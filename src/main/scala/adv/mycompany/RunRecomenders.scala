package adv.mycompany

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.expressions.Alias
import org.apache.spark.sql.functions.{lit, max, min}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}

import scala.util.Random

object RunRecomenders {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .appName("Intro")
      .config("spark.master", "local")
      .getOrCreate

    val rawUserArtistData = spark.read.textFile("E:\\vivek\\data\\profiledata_06-May-2005.tar\\profiledata_06-May-2005\\user_artist_data.txt")
    val rawArtistData = spark.read.textFile("E:\\vivek\\data\\profiledata_06-May-2005.tar\\profiledata_06-May-2005\\artist_data.txt")
    val rawArtistAlias= spark.read.textFile("E:\\vivek\\data\\profiledata_06-May-2005.tar\\profiledata_06-May-2005\\artist_alias.txt")

    val runRecomenders=new RunRecomenders(spark)
    runRecomenders.preparation(rawUserArtistData,rawArtistData,rawArtistAlias)
    runRecomenders.model(rawUserArtistData,rawArtistData,rawArtistAlias)



  }
}


class RunRecomenders(private val spark: SparkSession) {



  import spark.implicits._;



  def preparation(rawUserArtistData: Dataset[String],
                  rawArtistData:Dataset[String],
                  rawArtistAlias:Dataset[String]
                 ) :Unit ={
    rawUserArtistData.take(5).foreach(println);


    val userArtistDF = rawUserArtistData.map { line =>
      val Array(user, artist, _*) = line.split(' ')
      (user.toInt, artist.toInt)
    }.toDF("user", "artist")

    userArtistDF.agg(min("user"), max("user"), min("user"), max("artist")).show(false);

    val artistById= buildArtisById(rawArtistData)
    val artistAlias=buildArtisAlias(rawArtistAlias)

    artistById.filter($"id" isin(1208690,1003926)).show()


  }




  def model(rawUserArtistData: Dataset[String], rawArtistData: Dataset[String], rawArtistAlias: Dataset[String]): Unit = {
      val bartistData= spark.sparkContext.broadcast(buildArtisAlias(rawArtistAlias))

      val trainData = buildCount(rawUserArtistData,bartistData)
    trainData.show(false)
    println("========")
      val model=new ALS().
        setSeed(Random.nextLong())
        .setImplicitPrefs(true)
        .setRank(10)
        .setRegParam(0.01)
        .setAlpha(1.0)
        .setMaxIter(5)
        .setUserCol("user")
        .setItemCol("artist")
        .setRatingCol("count")
        .setPredictionCol("predictions")
        .fit(trainData)

    trainData.unpersist()
    model.userFactors.select("features").show(false)

    val userId= 2093760
    val existingArtistId=trainData.filter($"user"=== userId)
      .select("artist")
      .as[Int]
      .collect()
    val artistById=buildArtisById(rawArtistData)
    artistById.filter($"id" isin(existingArtistId:_*)).show(false)

    val topRecommnedations=makeRecommendation(model,userId,5)

    val recommnedationList=topRecommnedations.select("artist").as[Int].collect()

    artistById.filter($"id" isin(recommnedationList:_*)).show(false)


    model.userFactors.unpersist()
    model.itemFactors.unpersist()
  }
  def makeRecommendation(model: ALSModel, userId: Int, i: Int) = {
    val toRecommend=model.itemFactors.select($"id".as("artist")).withColumn("user", lit(userId))
    model.transform(toRecommend).select("artist","predictions").orderBy($"predictions".desc).limit(i)
  }

  def buildArtisById(rawArtistData: Dataset[String]): DataFrame = {

     rawArtistData.flatMap { line =>
      val (id,name)=line.span(_ !='\t')
      if(name.isEmpty){
        None
      }else{
        try {
          Some((id.toInt, name.trim))
        }
        catch {
          case _:NumberFormatException=>None
        }
      }

  }.toDF("id","name")

  }

  def buildCount(rawUserArtistData: Dataset[String], bartistData: Broadcast[Map[Int, Int]]) = {
    rawUserArtistData.map{line =>
    val Array(userId,artistId,count)= line.split(" ").map(_.toInt)
    val finalArtistId=bartistData.value.getOrElse(artistId,artistId)
    (userId,finalArtistId,count)
    }toDF("user", "artist", "count")
  }

  def buildArtisAlias(rawArtistAlias: Dataset[String]):Map[Int,Int] = {
    rawArtistAlias.flatMap{ line =>{
      val Array(artist,alias)= line.split('\t')
      if(artist.isEmpty){
        None
      }else{
        try {
          Some((artist.toInt,alias.toInt))
        }
        catch {
          case _:NumberFormatException=>None
        }
      }
    } }.collect().toMap


  }

}