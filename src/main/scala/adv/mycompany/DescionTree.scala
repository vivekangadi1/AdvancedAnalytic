package adv.mycompany

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Random

object DescionTree {
  def main(args :Array[String]):Unit ={

    val spark=SparkSession.builder()
              .appName("Descion Tree")
              .config("spark.master","local[*]")
              .getOrCreate()
    import spark.implicits._
    val dataWithoutHeader= spark.read
      .option("inferSchema","true")
      .option("header","false").csv("E:\\vivek\\data\\desciontree\\covtype.data")

    dataWithoutHeader.show(5)

    val colNames = Seq(
      "Elevation", "Aspect", "Slope",
      "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
      "Horizontal_Distance_To_Roadways",
      "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
      "Horizontal_Distance_To_Fire_Points"
    ) ++ (
      (0 until 4).map(i => s"Wilderness_Area_$i")
      ) ++ (
      (0 until 40).map(i => s"Soil_Type_$i")
      ) ++ Seq("Cover_Type")

    val data=dataWithoutHeader.toDF(colNames:_*).withColumn("Cover_Type", $"Cover_Type".cast("double"))

    data.show()
    data.head
    val Array(trainData,testData)=data.randomSplit(Array(0.9,0.1))

    trainData.cache()
    testData.cache()

    val dec=new DescionT(spark)

   // dec.simpleDesciontree(trainData,testData)
//    dec.randomClassifier(trainData,testData)
    dec.evaluation(trainData,testData)

  }

}

class DescionT(private  val spark:SparkSession){

  import spark.implicits._

  def simpleDesciontree(trainData:DataFrame,testData:DataFrame): Unit ={
    val inputCols=trainData.columns.filter(_ !="Cover_Type")

    val assemble=new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("Features")

    val assembledTrainData =assemble.transform(trainData)

    assembledTrainData.select("Features").show(5)


    val classifier =new DecisionTreeClassifier().
      setSeed(Random.nextLong())
        .setLabelCol("Cover_Type")
        .setFeaturesCol("Features")
        .setPredictionCol("Prediction")

    val model = classifier.fit(assembledTrainData)
    println(model.toDebugString)

    model.featureImportances.toArray.zip(inputCols).
      sorted.reverse.foreach(println)

    val prediction = model.transform(assembledTrainData)

    prediction.select("Cover_Type","Prediction","probability").show(5)

    val evaluator=new MulticlassClassificationEvaluator().setLabelCol("Cover_Type").setPredictionCol("Prediction")

    val accuracy=evaluator.setMetricName("accuracy").evaluate(prediction)

    val f1Accuracy=evaluator.setMetricName("f1").evaluate(prediction)

    println(accuracy)
    println(f1Accuracy+"55222921"+"0.0345"+"0.0461")

  }

  def classProbabilities(data: DataFrame): Array[Double] = {
    val total = data.count()
    data.groupBy("Cover_Type").count().
      orderBy("Cover_Type").
      select("count").as[Double].
      map(_ / total).
      collect()
    data.groupBy("Cover_Type").count().
      orderBy("Cover_Type").
      select("count").as[Double].
      map(_ / total).
      collect()
  }

  def randomClassifier(trainData: DataFrame, testData: DataFrame): Unit = {
    val trainPriorProbabilities = classProbabilities(trainData)
    val testPriorProbabilities = classProbabilities(testData)
    val accuracy = trainPriorProbabilities.zip(testPriorProbabilities).map {
      case (trainProb, cvProb) => println(trainProb, cvProb);trainProb * cvProb
    }.sum

    println(accuracy)
  }

  def evaluation(trainData: DataFrame, testData: DataFrame): Unit ={
    val inputcol =trainData.columns.filter(_ !="Cover_Type")

    val assembler =new VectorAssembler()
      .setInputCols(inputcol)
      .setOutputCol("featuresVectors")

    val classifier = new DecisionTreeClassifier()
      .setLabelCol("Cover_Type")
      .setFeaturesCol("featuresVectors")
      .setPredictionCol("predictions")

    val pipeline =new Pipeline().setStages(Array(assembler,classifier))

    val paramGrid=new ParamGridBuilder()
      .addGrid(classifier.impurity,Seq("gini","entropy"))
      .addGrid(classifier.maxDepth,Seq(1,20))
      .addGrid(classifier.maxBins,Seq(40,300))
      .addGrid(classifier.minInfoGain,Seq(0.0,0.1))
      .build()

    val multiclasseval=new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
      .setLabelCol("Cover_Type")
      .setPredictionCol("predictions")

    val trainValid=new TrainValidationSplit()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(multiclasseval)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.9)

    val validatorModel= trainValid.fit(trainData)

    val paramsAndMetrics = validatorModel.validationMetrics.
      zip(validatorModel.getEstimatorParamMaps).sortBy(-_._1)

    paramsAndMetrics.foreach { case (metric, params) =>
      println(metric)
      println(params)
      println()
    }

    val bestModel = validatorModel.bestModel

    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    println(validatorModel.validationMetrics.max)

    val testAccuracy = multiclasseval.evaluate(bestModel.transform(testData))
    println(testAccuracy)

    val trainAccuracy = multiclasseval.evaluate(bestModel.transform(trainData))
    println(trainAccuracy)


  }

  def unencodeOneHot(data: DataFrame): DataFrame = {
    val wildernessCols = (0 until 4).map(i => s"Wilderness_Area_$i").toArray

    val wildernessAssembler = new VectorAssembler().
      setInputCols(wildernessCols).
      setOutputCol("wilderness")

    val unhotUDF = udf((vec: Vector) => vec.toArray.indexOf(1.0).toDouble)

    val withWilderness = wildernessAssembler.transform(data).
      drop(wildernessCols:_*).
      withColumn("wilderness", unhotUDF($"wilderness"))

    val soilCols = (0 until 40).map(i => s"Soil_Type_$i").toArray

    val soilAssembler = new VectorAssembler().
      setInputCols(soilCols).
      setOutputCol("soil")

    soilAssembler.transform(withWilderness).
      drop(soilCols:_*).
      withColumn("soil", unhotUDF($"soil"))
  }

  def evaluateCategorical(trainData: DataFrame, testData: DataFrame): Unit = {
    val unencTrainData = unencodeOneHot(trainData)
    val unencTestData = unencodeOneHot(testData)

    val inputCols = unencTrainData.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")

    val indexer = new VectorIndexer().
      setMaxCategories(40).
      setInputCol("featureVector").
      setOutputCol("indexedVector")

    val classifier = new DecisionTreeClassifier().
      setSeed(Random.nextLong()).
      setLabelCol("Cover_Type").
      setFeaturesCol("indexedVector").
      setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, indexer, classifier))

    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.impurity, Seq("gini", "entropy")).
      addGrid(classifier.maxDepth, Seq(1, 20)).
      addGrid(classifier.maxBins, Seq(40, 300)).
      addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).
      build()

    val multiclassEval = new MulticlassClassificationEvaluator().
      setLabelCol("Cover_Type").
      setPredictionCol("prediction").
      setMetricName("accuracy")

    val validator = new TrainValidationSplit().
      setSeed(Random.nextLong()).
      setEstimator(pipeline).
      setEvaluator(multiclassEval).
      setEstimatorParamMaps(paramGrid).
      setTrainRatio(0.9)

    val validatorModel = validator.fit(unencTrainData)

    val bestModel = validatorModel.bestModel

    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    val testAccuracy = multiclassEval.evaluate(bestModel.transform(unencTestData))
    println(testAccuracy)
  }
}
