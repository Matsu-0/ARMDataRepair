package Algorithm;


import Algorithm.util.KDTreeUtil;
import Algorithm.util.TimeSeriesPredictor;
import Algorithm.util.VARUtil;

import java.util.ArrayList;

public class MasterDetector {
    private final double[][] td;
    private double[][] td_repaired;
    private double[][] td_prediction;  // store predicted values
    private boolean[] td_anomalies;
    //    private boolean[] anomalies_in_repaired;
    private final KDTreeUtil kdTreeUtil;
    private final long[] td_time;
    private final int columnCnt;
    private final int k;
    private final int p;
    private double[] std;
    private int[] initial_window;
    private double regression_loss;
    private double prediction_regression_loss;  // regression loss of predictions
    private int n_repair_loss_count;    // count of repair loss accumulations, for normalization
    private int n_prediction_loss_count;  // count of prediction loss accumulations, for normalization

    private TimeSeriesPredictor prediction_model;

    private final int n;

    private double eta;
    private final long cost_time;
    private long prediction_time;  // time spent in prediction phase
    private long master_repair_time;  // time spent in master data repair phase

    public MasterDetector(double[][] td, KDTreeUtil kdTreeUtil, long[] td_time, int columnCnt, int k, int p, double eta) {
        this(td, kdTreeUtil, td_time, columnCnt, k, p, eta, new VARUtil(columnCnt));
    }
    
    public MasterDetector(double[][] td, KDTreeUtil kdTreeUtil, long[] td_time, int columnCnt, int k, int p, double eta, TimeSeriesPredictor predictor) {
        this.td = td;
        this.kdTreeUtil = kdTreeUtil;
        this.td_time = td_time;
        this.columnCnt = columnCnt;
        this.k = k;
        this.p = p;
        this.eta = eta;
        this.n = td.length;
        this.prediction_model = predictor;
        long startTime = System.currentTimeMillis();
//        this.testModelOnly(0.8);
        this.repair();
        long endTime = System.currentTimeMillis();
        this.cost_time = endTime - startTime;
        System.out.println("MasterRepair time cost:" + cost_time + "ms (prediction: " + prediction_time + "ms, master repair: " + master_repair_time + "ms)");
    }

    //    public double delta(double[] t_tuple, double[] m_tuple) {
//        double distance = 0d;
//        for (int pos = 0; pos < columnCnt; pos++) {
//            double temp = t_tuple[pos] - m_tuple[pos];
//            temp = temp / std[pos];
//            distance += temp * temp;
//        }
//        distance = Math.sqrt(distance);
//        return distance;
//    }
    public double delta(double[] t_tuple, double[] m_tuple) {
        double distance = 0d;
        for (int pos = 0; pos < columnCnt; pos++) {
            double temp = t_tuple[pos] - m_tuple[pos];
            temp = temp / std[pos];
            distance += temp;
        }
//        distance = Math.sqrt(distance);
        return distance;
    }

    private double varianceImperative(double[] value) {
        double average = 0.0;
        int cnt = 0;
        for (double p : value) {
            if (!Double.isNaN(p)) {
                cnt += 1;
                average += p;
            }
        }
        if (cnt == 0) {
            return 0d;
        }
        average /= cnt;

        double variance = 0.0;
        for (double p : value) {
            if (!Double.isNaN(p)) {
                variance += (p - average) * (p - average);
            }
        }
        return variance / cnt;
    }

    private double[] getColumn(int pos) {
        double[] column = new double[n];
        for (int i = 0; i < n; i++) {
            column[i] = this.td[i][pos];
        }
        return column;
    }

    public void call_std() {
        this.std = new double[this.columnCnt];
        for (int i = 0; i < this.columnCnt; i++) {
            std[i] = Math.sqrt(varianceImperative(getColumn(i)));
        }
    }

    public boolean checkConsistency(double[] tuple) {
        double[] NN = kdTreeUtil.nearestNeighbor(tuple);
        double delta = delta(tuple, NN);
//        System.out.println(Arrays.toString(tuple) + Arrays.toString(NN));
//
//        System.out.println(delta);
        if (delta > eta) {
            return false;
        } else return true;
    }

    public void getOriginalAnomaliesAndLearnModel() {
//        int left = 0;
//        int right = 0;
//        int m = 10;
        ArrayList<ArrayList<Double>> learning_samples = new ArrayList<>();
//        int samples_cnt = 0;
        td_anomalies = new boolean[n];
        for (int i = 0; i < td.length; i++) {
            double[] tuple = td[i];
            boolean isNormal = checkConsistency(tuple);
            td_anomalies[i] = !isNormal;

//            if (samples_cnt < m) {
//                if (right - left + 1 == p) {
//                    samples_cnt++;
//                    for (int j = left; j <= right; j++) {
//                        ArrayList<Double> sample = new ArrayList<>();
//                        for (double value : td[j]) {
//                            sample.add(value);
//                        }
//                        learning_samples.add(sample);
//                    }
//                }
//            }
            if (isNormal) {
                ArrayList<Double> sample = new ArrayList<>();
                for (double value : tuple) {
                    sample.add(value);
                }
                learning_samples.add(sample);
//                right++;
            }
//            else {
//                left = i + 1;
//                right = i + 1;
//            }
        }
        // if prediction_model is not initialized, use default VAR model
        if (this.prediction_model == null) {
            this.prediction_model = new VARUtil(columnCnt);
        }
        this.prediction_model.fit(learning_samples);
    }

    //
    public void findInitialWindow(int p) {
        initial_window = new int[2];
        int left = 0;
        int right = 0;
        for (int i = 0; i < td_anomalies.length; i++) {
            if (right - left + 1 == p) {
                initial_window[0] = left;
                initial_window[1] = right;
                break;
            }
            if (td_anomalies[i] == Boolean.TRUE) {
                left = i + 1;
                right = i + 1;
            } else {
                right++;
            }
        }
    }

    public double[][] getWindow(double[][] data, int i, int p) {
        if (i < p) {
            System.out.println("ERROR: i must be greater than p.");
            return null;
        }
        double[][] W = new double[p][columnCnt];
        System.arraycopy(data, i - p, W, 0, p);
        return W;
    }
public void forwardRepairing(int p) {
        int i = initial_window[1] + 1;

        while (i < n) {
            if (td_anomalies[i] == Boolean.TRUE) {
                double[][] W_repaired = getWindow(this.td_repaired, i, p);
                
                // prediction phase (record time)
                long predStart = System.currentTimeMillis();
                double[] x_repaired_predicted = arrayToList(prediction_model.predict(W_repaired));
                long predEnd = System.currentTimeMillis();
                prediction_time += (predEnd - predStart);
                
                // store predicted value
                this.td_prediction[i] = x_repaired_predicted.clone();
                
                // master data repair phase (record time)
                long repairStart = System.currentTimeMillis();
                double[][] candidates =
                        this.kdTreeUtil.kNearestNeighbors(x_repaired_predicted, this.k);
                //        find the optimal repair
                double[] optimal_repair = new double[columnCnt];
                double min_dis = Double.MAX_VALUE;
                for (double[] candidate : candidates) {
                    if (delta(x_repaired_predicted, candidate) < min_dis) {
                        min_dis = delta(x_repaired_predicted, candidate);
                        optimal_repair = candidate;
                    }
                }
                long repairEnd = System.currentTimeMillis();
                master_repair_time += (repairEnd - repairStart);
                
                this.td_repaired[i] = optimal_repair;
                regression_loss += delta(optimal_repair, x_repaired_predicted);
                prediction_regression_loss += delta(x_repaired_predicted, td[i]);
                n_repair_loss_count++;
                n_prediction_loss_count++;
            } else {
                // normal point: predicted value equals original value
                this.td_prediction[i] = td[i].clone();
                this.td_repaired[i] = td[i];
            }
            i++;
        }
    }


    public void backwardRepairing(int p) {
        int i = initial_window[0] - 1;
        if (i < 0) {
            return;
        }

        while (i >= 0) {
            if (td_anomalies[i] == Boolean.TRUE) {
                double[] optimal_repair = new double[columnCnt];
                double[][] candidates =
                        this.kdTreeUtil.kNearestNeighbors(this.td_repaired[i + 1], k);
                double[][] W_repaired = getWindow(this.td_repaired, i + p, p);
                double[] x_repaired = this.td_repaired[i];
                
                // prediction phase (record time)
                long predStart = System.currentTimeMillis();
                double min_dis = Double.MAX_VALUE;
                double regression_loss_delta = 0.0;
                double[] best_prediction = null;
                
                for (double[] candidate : candidates) {
                    W_repaired[0] = candidate;
                    double[] x_repaired_predicted = arrayToList(prediction_model.predict(W_repaired));

                    if (delta(x_repaired_predicted, x_repaired) < min_dis) {
                        min_dis = delta(x_repaired_predicted, x_repaired);
                        regression_loss_delta = min_dis;
                        optimal_repair = candidate;
                        best_prediction = x_repaired_predicted.clone();
                    }
                }
                long predEnd = System.currentTimeMillis();
                prediction_time += (predEnd - predStart);
                
                // store predicted value
                if (best_prediction != null) {
                    this.td_prediction[i] = best_prediction;
                }
                
                // master data repair phase (record time)
                long repairStart = System.currentTimeMillis();
                // candidate search already done in prediction phase; here we mainly select the best value
                long repairEnd = System.currentTimeMillis();
                master_repair_time += (repairEnd - repairStart);
                
                this.td_repaired[i] = optimal_repair;
                this.regression_loss += regression_loss_delta;
                if (best_prediction != null) {
                    prediction_regression_loss += delta(best_prediction, td[i]);
                    n_prediction_loss_count++;
                }
                n_repair_loss_count++;
            } else {
                // normal point: predicted value equals original value
                this.td_prediction[i] = td[i].clone();
                this.td_repaired[i] = td[i];
            }
            i--;
        }
    }

    public void initList() {
        this.regression_loss = 0.0;
        this.prediction_regression_loss = 0.0;
        this.n_repair_loss_count = 0;
        this.n_prediction_loss_count = 0;
        td_prediction = new double[n][columnCnt];
        td_repaired = new double[n][columnCnt];
        this.prediction_time = 0;
        this.master_repair_time = 0;
//        anomalies_in_repaired = new boolean[n];
    }

    public void repair() {
        call_std();
        getOriginalAnomaliesAndLearnModel();
        findInitialWindow(p);
        initList();
        System.arraycopy(td, 0, td_repaired, 0, initial_window[1] + 1);
        backwardRepairing(p);
        forwardRepairing(p);
    }


    public void testModelOnly(double rate) {
        call_std();

        ArrayList<ArrayList<Double>> learning_samples = new ArrayList<>();
        td_anomalies = new boolean[n];
        this.regression_loss = 0.0;
        this.td_repaired = new double[n][columnCnt];
        for (int i = 0; i < td.length; i++) {
            double[] tuple = td[i];
            boolean isNormal = checkConsistency(tuple);
            td_anomalies[i] = !isNormal;
        }
        int train_window = (int) (n * rate);
        for (int i = 0; i < train_window; i++) {
            ArrayList<Double> sample = new ArrayList<>();
            for (double value : td[i]) {
                sample.add(value);
            }
            learning_samples.add(sample);
        }
        // if prediction_model is not initialized, use default VAR model
        if (this.prediction_model == null) {
            this.prediction_model = new VARUtil(columnCnt);
        }
        this.prediction_model.fit(learning_samples);

        for (int i = train_window; i < td.length; i++) {
            double[][] W = getWindow(td, i, p);
            ArrayList<Double> prediction = this.prediction_model.predict(W);
//            System.out.println(Arrays.toString(arrayToList(prediction)));
//            System.out.println(Arrays.toString(td[i]));
            this.regression_loss += delta(arrayToList(prediction), td[i]);
        }

        findInitialWindow(p);

        int i = initial_window[1] + 1;

        while (i < n) {
            if (td_anomalies[i] == Boolean.TRUE) {
                double[][] W_repaired = getWindow(this.td_repaired, i, p);
                double[] x_repaired_predicted = arrayToList(prediction_model.predict(W_repaired));
                this.td_repaired[i] = x_repaired_predicted;
            } else {
                this.td_repaired[i] = td[i];
            }
            i++;
        }
    }

    public double[] arrayToList(ArrayList<Double> arrayList) {
        double[] list = new double[arrayList.size()];
        for (int i = 0; i < arrayList.size(); i++) {
            list[i] = arrayList.get(i);
        }
        return list;
    }

    public double[][] getTd_repaired() {
        return td_repaired;
    }
    
    public double[][] getTd_prediction() {
        return td_prediction;
    }

//    public boolean[] getAnomalies_in_repaired() {
//        return anomalies_in_repaired;
//    }

    public long getCost_time() {
        return cost_time;
    }
    
    public long getPrediction_time() {
        return prediction_time;
    }
    
    public long getMaster_repair_time() {
        return master_repair_time;
    }
    
    public long getTotal_repair_time() {
        return prediction_time + master_repair_time;
    }

    public double getRegression_loss() {
        return regression_loss;
    }
    
    public double getPrediction_regression_loss() {
        return prediction_regression_loss;
    }

    /** Normalized repair regression loss (divided by anomaly count to reduce scale effect) */
    public double getRegression_loss_normalized() {
        return n_repair_loss_count <= 0 ? 0.0 : regression_loss / n_repair_loss_count;
    }

    /** Normalized prediction regression loss (divided by anomaly count to reduce scale effect) */
    public double getPrediction_regression_loss_normalized() {
        return n_prediction_loss_count <= 0 ? 0.0 : prediction_regression_loss / n_prediction_loss_count;
    }
}