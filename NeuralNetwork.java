import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

/**
 * models a simple ANN to solve the Titanic problem
 * 
 * @author Vance Spears
 * @version 4/21/23
 */
public class NeuralNetwork {
   /** number of features in the input file */
   public static final int NUM_FEATS_READ = 13;

   /** number of features in the model */
   public static final int NUM_FEATS = 9;

   /** multiple of inputs to use for number of hidden nodes */
   public static final double HIDDEN_PCT = 10;

   /** number of output nodes */
   public static final int NUM_OUTPUTS = 1;

   /** number of training epochs */
   public static final int NUM_EPOCHS = 10000;

   /** rate at which weights are adjusted */
   public static final double LEARN_RATE = 1.0;

   /** array of passenger data */
   private ArrayList<Passenger> passengers;

   /** feature set of maxium age, num parents/children, and num siblings/spouses */
   private FeatureSet myMaxes = new FeatureSet();

   private double[] inputs;

   private double[][] weights_input;

   private double[] hidden;

   private double[][] weights_hidden;

   private double[] outputs;

   /**
    * default constructor
    */
   public NeuralNetwork() {

   }

   /**
    * reads data from input file
    * 
    * @param filename the file to read
    * @return a list of Passenger objects
    */
   public ArrayList<Passenger> read_passengers(String filename) {
      passengers = new ArrayList<Passenger>();

      try {
         BufferedReader reader = new BufferedReader(new FileReader(filename));

         String line;
         line = reader.readLine();
         line = reader.readLine();
         while (line != null) {
            String[] read = line.split(",");

            String[] parts = new String[NUM_FEATS_READ];
            for (int idx = 0; idx < read.length; idx++) {
               if (read[idx].equals("")) {
                  parts[idx] = "-1";
               } else {
                  parts[idx] = read[idx];
               }
            }

            if (read.length == NUM_FEATS_READ - 1) {
               parts[NUM_FEATS_READ - 1] = "-1";
            } else {
               parts[NUM_FEATS_READ - 1] = read[NUM_FEATS_READ - 1];
            }

            Passenger p = new Passenger();
            p.id = Integer.parseInt(parts[0]);
            p.survived = Integer.parseInt(parts[1]);
            p.pclass = Integer.parseInt(parts[2]);
            p.name = parts[3] + "," + parts[4];

            if (parts[5].equals("male")) {
               p.sex = Passenger.MALE;
            } else {
               p.sex = Passenger.FEMALE;
            }

            p.age = Double.parseDouble(parts[6]);
            p.sibsp = Integer.parseInt(parts[7]);
            p.parch = Integer.parseInt(parts[8]);
            p.ticket = parts[9];
            p.fare = Double.parseDouble(parts[10]);
            p.cabin = parts[11];

            if (parts[12].equals("S")) {
               p.embarked = Passenger.SOUTHAMPTON;
            } else if (parts[12].equals("C")) {
               p.embarked = Passenger.CHERBOURG;
            } else if (parts[12].equals("Q")) {
               p.embarked = Passenger.QUEENSLAND;
            }

            passengers.add(p);

            line = reader.readLine();
         }

         reader.close();

      } catch (IOException ioe) {
         System.err.println(ioe);
         System.exit(1);
      }

      return passengers;
   }

   /**
    * runs sigmoid function for forward propagation
    * 
    * @param x the input value
    * @return sigmoid(x)
    */
   public double sigmoid(double x) {
      return 1 / (1 + Math.exp(x * -1));
   }

   /**
    * runs sigmoid derivative function for backward propagation
    * 
    * @param x the input value
    * @return sigmoid derivative of x
    */
   public double sigmoid_derivative(double x) {
      return x * (1 - x);
   }

   /**
    * runs step function for a given value
    * 
    * @param x the input value
    * @return step function value for x
    */
   public int step_function(double x) {
      if (x < 0) {
         return 0;
      } else {
         return 1;
      }
   }

   /**
    * runs hard max on a set of doubles
    * 
    * @param vals the set of values
    * @return int vector with hard max applied to vals
    */
   public int[] hardMax(double[] vals) {
      int maxIdx = 0;
      double maxVal = vals[0];

      for (int idx = 1; idx < vals.length; idx++) {
         if (vals[idx] > maxVal) {
            maxIdx = idx;
         }
      }

      int outs[] = new int[vals.length];
      outs[maxIdx] = 1;

      return outs;
   }

   /**
    * rounds a double to two decimal places
    * 
    * @param val the input value
    * @return the rounded value
    */
   public double round2(double val) {
      return (Math.round(val * 100.0) / 100.0);
   }

   /**
    * extracts features from the input data
    * 
    * @return array of input elements
    */
   public FeatureSet[] get_feature_data() {
      FeatureSet[] fs = new FeatureSet[passengers.size()];

      for (int i = 0; i < passengers.size(); i++) {
         double[] features = new double[NUM_FEATS];

         features[0] = passengers.get(i).age;
         features[1] = passengers.get(i).sibsp;
         features[2] = passengers.get(i).parch;
         if (passengers.get(i).sex == Passenger.MALE) {
            features[3] = 1;
            features[4] = 0;
         } else if (passengers.get(i).sex == Passenger.FEMALE) {
            features[3] = 0;
            features[4] = 1;
         }
         features[5] = passengers.get(i).pclass;
         if (passengers.get(i).embarked == Passenger.SOUTHAMPTON) {
            features[6] = 1;
            features[7] = 0;
            features[8] = 0;
         } else if (passengers.get(i).embarked == Passenger.CHERBOURG) {
            features[6] = 0;
            features[7] = 1;
            features[8] = 0;
         } else if (passengers.get(i).embarked == Passenger.QUEENSLAND) {
            features[6] = 0;
            features[7] = 0;
            features[8] = 1;
         }

         fs[i] = new FeatureSet(features);
      }
      return fs;
   }

   /**
    * extracts labels from input data
    * 
    * @return array of output vectors
    */
   public int[][] get_labels() {
      int[][] labels = new int[passengers.size()][1];
      for (int i = 0; i < passengers.size(); i++) {
         labels[i][0] = passengers.get(i).survived;
      }
      return labels;
   }

   /**
    * finds max vals for age, parents, siblings
    */
   public void find_maxes() {
      for (Passenger p : passengers) {
         if (p.age > myMaxes.features[0]) {
            myMaxes.features[0] = p.age;
         }
         if (p.sibsp > myMaxes.features[1]) {
            myMaxes.features[1] = p.sibsp;
         }
         if (p.parch > myMaxes.features[2]) {
            myMaxes.features[2] = p.parch;
         }
      }
   }

   /** calculates the mean age of passengers */
   public double get_mean_age() {
      double ageSum = 0;
      double ageCount = 0;
      for (Passenger p : passengers) {
         if (p.age > 0) {
            ageSum += p.age;
            ageCount++;
         }
      }

      return ageSum / ageCount;
   }

   /**
    * scales the data so each feature is in [0.0, 1.0]
    * 
    * @param data the data to scale
    * @return array with scaled data
    */
   public FeatureSet[] scale_dataset(FeatureSet[] data) {
      FeatureSet[] scaledData = new FeatureSet[data.length];

      for (int i = 0; i < data.length; i++) {
         FeatureSet fs = data[i];
         FeatureSet scaledFs = new FeatureSet(fs.features.clone());

         if (fs.features[0] > 0) {
            scaledFs.features[0] = fs.features[0] / myMaxes.features[0];
         } else {
            scaledFs.features[0] = get_mean_age() / myMaxes.features[0];
         }

         scaledFs.features[1] = fs.features[1] / myMaxes.features[1];
         scaledFs.features[2] = fs.features[2] / myMaxes.features[2];
         scaledFs.features[5] = 1 / fs.features[5];

         scaledData[i] = scaledFs;
      }

      return scaledData;
   }

   /**
    * sets up the network structure
    */
   public void setup_network() {
      final int NUM_HIDDEN = (int) (NUM_FEATS * HIDDEN_PCT);
      Random random = new Random();

      inputs = new double[NUM_FEATS];
      hidden = new double[NUM_HIDDEN];
      outputs = new double[NUM_OUTPUTS];

      weights_input = new double[NUM_FEATS][NUM_HIDDEN];
      for (int i = 0; i < NUM_FEATS; i++) {
         for (int j = 0; j < NUM_HIDDEN; j++) {
            weights_input[i][j] = random.nextGaussian();
         }
      }

      weights_hidden = new double[NUM_HIDDEN][NUM_OUTPUTS];
      for (int i = 0; i < NUM_HIDDEN; i++) {
         for (int j = 0; j < NUM_OUTPUTS; j++) {
            weights_hidden[i][j] = random.nextGaussian();
         }
      }
   }

   /**
    * runs forward propagation algorithm for a given element
    * 
    * @param data the element to process
    * @return the vector of output values
    */
   public double[] forward_propagation(FeatureSet data) {
      final int NUM_HIDDEN = (int) (NUM_FEATS * HIDDEN_PCT);

      for (int i = 0; i < NUM_HIDDEN; i++) {
         double hidden_weighted_sum = 0;
         for (int j = 0; j < NUM_FEATS; j++) {
            hidden_weighted_sum += data.features[j] * weights_input[j][i];
         }
         hidden[i] = sigmoid(hidden_weighted_sum);
      }

      for (int i = 0; i < NUM_OUTPUTS; i++) {
         double output_weighted_sum = 0;
         for (int j = 0; j < NUM_HIDDEN; j++) {
            output_weighted_sum += hidden[j] * weights_hidden[j][i];
         }
         outputs[i] = sigmoid(output_weighted_sum);
      }

      return outputs;
   }

   /**
    * runs back propagation algorithm to update weights
    * 
    * @param label_set the labels to use for cost calculation
    */
   public void back_propagation(int[] label_set) {
      for (int i = 0; i < outputs.length; i++) {
         double output_error = sigmoid_derivative(outputs[i]) * (label_set[i] - outputs[i]);
         double hiddenErrors[] = new double[hidden.length];
         for (int j = 0; j < hidden.length; j++) {
            hiddenErrors[j] = weights_hidden[j][i] * output_error * sigmoid_derivative(hidden[j]);
            weights_hidden[j][i] = weights_hidden[j][i] + (LEARN_RATE * hidden[j] * output_error);
            for (int k = 0; k < inputs.length; k++) {
               weights_input[k][j] = weights_input[k][j] + (LEARN_RATE * inputs[k] * hiddenErrors[j]);
            }
         }
      }
   }

   /**
    * runs the training algorithm on a network
    * 
    * @param epochs   the number of epochs to train
    * @param filename the input file of training data
    */
   public void train_neural_network(int epochs, String filename) {
      setup_network();
      read_passengers(filename);
      find_maxes();
      FeatureSet[] feature_data = get_feature_data();
      FeatureSet[] scaled_feature_data = scale_dataset(feature_data);
      int[][] labels = get_labels();
      for (int i = 0; i < epochs; i++) {
         int matches = 0;
         for (int j = 0; j < scaled_feature_data.length - 1; j++) {
            outputs = forward_propagation(scaled_feature_data[j]);
            if (Math.round(outputs[0]) == labels[j][0]) {
               matches++;
            }
            back_propagation(labels[j]);
         }
         System.out.println(100 * (double) matches / scaled_feature_data.length);
      }
   }

   /**
    * runs the test algorithm on a network
    * 
    * @param filename the input file of training data
    */
   public void test_neural_network(String filename) {
      read_passengers(filename);
      find_maxes();
      FeatureSet[] feature_data = get_feature_data();
      FeatureSet[] scaled_feature_data = scale_dataset(feature_data);
      int[][] labels = get_labels();
      int matches = 0;
      for (int j = 0; j < scaled_feature_data.length - 1; j++) {
         outputs = forward_propagation(scaled_feature_data[j]);
         if (Math.round(outputs[0]) == labels[j][0]) {
            matches++;
         }
      }
      System.out.println("Success rate: " + round2(100 * (double) matches / scaled_feature_data.length) + "%");
   }

   /**
    * tests a single passenger on a network
    * 
    * @param passenger the passenger for testing
    */
   public void test_neural_network(Passenger passenger) {
      passengers = new ArrayList<>();
      passengers.add(passenger);
      find_maxes();
      FeatureSet[] feature_data = get_feature_data();
      FeatureSet[] scaled_feature_data = scale_dataset(feature_data);
      outputs = forward_propagation(scaled_feature_data[0]);
      System.out.println("Survival chance: " + round2(100 * outputs[0]) + "%");
   }

   public static void main(String[] args) {
      NeuralNetwork test = new NeuralNetwork();
      test.train_neural_network(NUM_EPOCHS, "./titanic/train.csv");

      Scanner reader = new Scanner(System.in);
      System.out.println("Training complete!");
      System.out.println();
      System.out.print("Select input mode (f for file, i for interactive): ");
      String input = reader.nextLine();

      if (input.toLowerCase().equals("f")) {
         test.test_neural_network("./titanic/test.csv");

      } else if (input.toLowerCase().equals("i")) {
         Passenger p = test.new Passenger();

         System.out.print("Enter age: ");
         p.age = Double.parseDouble(reader.nextLine());

         System.out.print("Enter # of siblings/spouses: ");
         p.sibsp = Integer.parseInt(reader.nextLine());

         System.out.print("Enter # of parents/children: ");
         p.parch = Integer.parseInt(reader.nextLine());

         System.out.print("Enter sex (M or F): ");
         String sex = reader.nextLine().toUpperCase();
         if (sex.equals("M")) {
            p.sex = Passenger.MALE;
         } else if (sex.equals("F")) {
            p.sex = Passenger.FEMALE;
         }

         System.out.print("Enter class: ");
         p.pclass = Integer.parseInt(reader.nextLine());

         System.out.print("Enter embarkation point (S, C, or Q): ");
         String embarked = reader.nextLine().toUpperCase();
         if (embarked.equals("S")) {
            p.embarked = Passenger.SOUTHAMPTON;
         } else if (embarked.equals("C")) {
            p.embarked = Passenger.CHERBOURG;
         } else if (embarked.equals("Q")) {
            p.embarked = Passenger.QUEENSLAND;
         }

         test.test_neural_network(p);
      }

      reader.close();
   }

   /**
    * stores info for each row in data file
    */
   private class Passenger {
      /** passenger died */
      public static final int DEAD = 0;

      /** passenger survived */
      public static final int ALIVE = 1;

      /** passenger is male */
      public static final int MALE = 0;

      /** passenger is female */
      public static final int FEMALE = 1;

      /** passenger embarked in Southampton */
      public static final int SOUTHAMPTON = 0;

      /** passenger embarked in Cherbourg */
      public static final int CHERBOURG = 1;

      /** passenger embarked in Queensland */
      public static final int QUEENSLAND = 2;

      /** array of toString values for survived */
      public static final String[] LIV_STRS = { "D", "A" };

      /** array of toString values for sex */
      public static final String[] SEX_STRS = { "M", "F" };

      /** array of toString values for embarked */
      public static final String[] EMB_STRS = { "S", "C", "Q" };

      /** unique id value for passenger */
      public int id;

      /** indicates whether passenger survived */
      public int survived;

      /** class of passenger (1, 2, 3) */
      public int pclass;

      /** name of passenger */
      public String name;

      /** sex of passenger (male or female) */
      public int sex;

      /** age of passenger */
      public double age;

      /** number of siblings or spouses for passenger */
      public int sibsp;

      /** number of parents or children for passenger */
      public int parch;

      /** ticket number of passenger */
      public String ticket;

      /** cost of fare for passenger */
      public double fare;

      /** cabin ID for passenger */
      public String cabin;

      /** location where passenger embarked */
      public int embarked;

      /**
       * converts passenger data to a string
       * 
       * @return string containing all properties
       */
      public String toString() {
         String out = LIV_STRS[survived] + " ";
         out += "#" + id;
         out += ": " + name + " ";
         out += "(" + SEX_STRS[sex] + "/" + age + ") ";
         out += "(" + pclass + "C, T=" + ticket + " F=" + fare + " C=" + cabin + " E=" + EMB_STRS[embarked] + ") ";
         out += "(S=" + sibsp + ", P=" + parch + ") ";

         return out;
      }
   }

   private class FeatureSet {

      /** array of feature labels */
      public static final String[] FEATURE_STRS = { "age", "# of siblings/spouses",
            "# of parents/children", "sex (M)", "sex (F)", "class", "embarkation point (S)",
            "embarkation point (C)", "embarkation point (Q)" };

      /** array of features */
      public double[] features;

      public FeatureSet() {
         features = new double[NUM_FEATS];
      }

      public FeatureSet(double[] inputFeatures) {
         if (inputFeatures.length == NUM_FEATS) {
            features = inputFeatures.clone();
         } else {
            System.err
                  .println("Error: FeatureSet inputFeatures length must be " + NUM_FEATS
                        + ", but an array with length " + inputFeatures.length
                        + " was passed. Exiting program.");
            System.exit(1);
         }
      }

      public String toString() {
         String out = "";
         for (int i = 0; i < NUM_FEATS; i++) {
            out += FEATURE_STRS[i] + ": " + features[i];
            if (i < NUM_FEATS - 1) {
               out += ", ";
            }
         }

         return out;
      }
   }
}