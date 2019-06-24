package com.kodcu.Spark_Bayes;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import scala.Function0;

public class Bayesanaliz {

	public static void main(String[] args) throws Exception {
		 
        SparkConf conf = new SparkConf()
                .setMaster("local")
                .setAppName("Naive Bayes Classifier")
                .set("spark.executor.memory", "1g");

        JavaSparkContext context = new JavaSparkContext(conf);

        RDD<LabeledPoint> trainData = MLUtils.loadLabeledData(context.sc(), "C:\\Users\\Samsung\\Desktop\\train.txt");

        NaiveBayesModel trained = NaiveBayes.train(trainData);


        Vector testData = Vectors.dense(new double[]{1, 1, 1, 0, 0, 0});
        double result = trained.predict(testData);


        System.out.println("Result = " + result);


    }
}
