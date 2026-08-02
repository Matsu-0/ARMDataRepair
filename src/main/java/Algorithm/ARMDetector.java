package Algorithm;


import Algorithm.util.KDTreeUtil;
import Algorithm.util.TimeSeriesPredictor;
import Algorithm.util.VARUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

public class ARMDetector {
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
    /** exclusive end index: model is fit only on conflict-free points in [0, modelTrainEnd) */
    private final int modelTrainEnd;

    private double eta;
    private final int t;  // max repair rounds
    private int actual_rounds;  // rounds actually executed
    private final long cost_time;
    private long model_train_time_ns;  // model.fit only
    private long prediction_time_ns;  // prediction phase (nanoseconds); test-only if modelTrainEnd < n
    private long repair_time_ns;  // domain-constraint repair phase; test-only if modelTrainEnd < n
    private long prediction_call_count;  // number of timed predict() calls
    private long wall_time;  // full repair() wall-clock time (ms)

    /** number of candidates kept for subsequent repair selection */
    private static final int CANDIDATE_TOP_N = 3;

    public ARMDetector(double[][] td, KDTreeUtil kdTreeUtil, long[] td_time, int columnCnt, int k, int p, double eta) {
        this(td, kdTreeUtil, td_time, columnCnt, k, p, eta, 3, new VARUtil(columnCnt), td.length);
    }

    public ARMDetector(double[][] td, KDTreeUtil kdTreeUtil, long[] td_time, int columnCnt, int k, int p, double eta, TimeSeriesPredictor predictor) {
        this(td, kdTreeUtil, td_time, columnCnt, k, p, eta, 3, predictor, td.length);
    }

    public ARMDetector(double[][] td, KDTreeUtil kdTreeUtil, long[] td_time, int columnCnt, int k, int p, double eta, int t) {
        this(td, kdTreeUtil, td_time, columnCnt, k, p, eta, t, new VARUtil(columnCnt), td.length);
    }

    public ARMDetector(double[][] td, KDTreeUtil kdTreeUtil, long[] td_time, int columnCnt, int k, int p, double eta, int t, TimeSeriesPredictor predictor) {
        this(td, kdTreeUtil, td_time, columnCnt, k, p, eta, t, predictor, td.length);
    }

    public ARMDetector(double[][] td, KDTreeUtil kdTreeUtil, long[] td_time, int columnCnt, int k, int p, double eta, int t, TimeSeriesPredictor predictor, int modelTrainEnd) {
        this.td = td;
        this.kdTreeUtil = kdTreeUtil;
        this.td_time = td_time;
        this.columnCnt = columnCnt;
        this.k = k;
        this.p = p;
        this.eta = eta;
        this.t = Math.max(1, t);
        this.n = td.length;
        this.modelTrainEnd = Math.max(p + 1, Math.min(modelTrainEnd, this.n));
        this.prediction_model = predictor;
        long wallStart = System.nanoTime();
//        this.testModelOnly(0.8);
        this.repair();
        this.wall_time = (System.nanoTime() - wallStart) / 1_000_000L;
        // reported cost is domain-constraint repair only (exclude detect/fit/predict)
        this.cost_time = getRepair_time();
        System.out.println("ARMRepair wall: " + wall_time + "ms, rounds: " + actual_rounds + "/" + this.t
                + " (train: " + getModel_train_time() + "ms, prediction: " + getPrediction_time()
                + "ms, avg pred: " + String.format("%.6f", getAvg_prediction_time_ms())
                + "ms, repair: " + getRepair_time() + "ms)"
                + ", modelTrainEnd=" + this.modelTrainEnd
                + (modelTrainEnd < n ? " [pred/repair timed on test only]" : ""));
    }

    /** When modelTrainEnd &lt; n (train/test split), only time indices in [modelTrainEnd, n). */
    private boolean shouldTimeIndex(int i) {
        return modelTrainEnd >= n || i >= modelTrainEnd;
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
        ArrayList<ArrayList<Double>> learning_samples = new ArrayList<>();
        td_anomalies = new boolean[n];
        for (int i = 0; i < td.length; i++) {
            double[] tuple = td[i];
            boolean isNormal = checkConsistency(tuple);
            td_anomalies[i] = !isNormal;
            // fit model only on conflict-free points inside the train prefix
            if (isNormal && i < modelTrainEnd) {
                ArrayList<Double> sample = new ArrayList<>();
                for (double value : tuple) {
                    sample.add(value);
                }
                learning_samples.add(sample);
            }
        }
        if (this.prediction_model == null) {
            this.prediction_model = new VARUtil(columnCnt);
        }
        long trainStart = System.nanoTime();
        this.prediction_model.fit(learning_samples);
        this.model_train_time_ns = System.nanoTime() - trainStart;
        System.out.println("ARM model trained on " + learning_samples.size()
                + " conflict-free points in [0, " + modelTrainEnd + ")"
                + ", trainTime=" + getModel_train_time() + "ms");
    }

    /**
     * Detect anomalies on the current repaired series. Returns anomaly count.
     */
    private int detectAnomaliesOnCurrent() {
        int count = 0;
        td_anomalies = new boolean[n];
        for (int i = 0; i < n; i++) {
            boolean isNormal = checkConsistency(td_repaired[i]);
            td_anomalies[i] = !isNormal;
            if (!isNormal) {
                count++;
            }
        }
        return count;
    }

    private int countAnomalies() {
        int count = 0;
        for (boolean flag : td_anomalies) {
            if (flag) {
                count++;
            }
        }
        return count;
    }

    // set initial window to the first p points at the beginning of the series
    public void findInitialWindow(int p) {
        initial_window = new int[2];
        initial_window[0] = 0;
        initial_window[1] = p - 1;
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

    /** non-negative normalized squared distance, used for candidate ranking */
    private double distanceAbs(double[] a, double[] b) {
        double distance = 0d;
        for (int pos = 0; pos < columnCnt; pos++) {
            double temp = (a[pos] - b[pos]) / std[pos];
            distance += temp * temp;
        }
        return distance;
    }

    private int findPrevNormal(int i) {
        for (int j = i - 1; j >= 0; j--) {
            if (!td_anomalies[j]) {
                return j;
            }
        }
        return -1;
    }

    private int findNextNormal(int i) {
        for (int j = i + 1; j < n; j++) {
            if (!td_anomalies[j]) {
                return j;
            }
        }
        return -1;
    }

    /**
     * Linear interpolation on nearby normal points (by timestamp if available, else by index).
     * Uses current repaired values for normal points.
     */
    private double[] linearInterpolateFromNormals(int i) {
        int prev = findPrevNormal(i);
        int next = findNextNormal(i);
        if (prev < 0 && next < 0) {
            return td_repaired[i].clone();
        }
        if (prev < 0) {
            return td_repaired[next].clone();
        }
        if (next < 0) {
            return td_repaired[prev].clone();
        }
        double ratio;
        if (td_time != null && td_time.length == n && td_time[next] != td_time[prev]) {
            ratio = (double) (td_time[i] - td_time[prev]) / (td_time[next] - td_time[prev]);
        } else {
            ratio = (double) (i - prev) / (next - prev);
        }
        double[] result = new double[columnCnt];
        for (int c = 0; c < columnCnt; c++) {
            result[c] = td_repaired[prev][c] + ratio * (td_repaired[next][c] - td_repaired[prev][c]);
        }
        return result;
    }

    /**
     * Three conventional interpolation candidates on normal points:
     * 1) linear interpolation, 2) previous normal (LOCF), 3) next normal (NOCB).
     */
    private double[][] getInterpolationCandidates(int i) {
        ArrayList<double[]> list = new ArrayList<>();
        int prev = findPrevNormal(i);
        int next = findNextNormal(i);

        list.add(linearInterpolateFromNormals(i));
        if (prev >= 0) {
            list.add(td_repaired[prev].clone());
        }
        if (next >= 0) {
            list.add(td_repaired[next].clone());
        }
        return list.toArray(new double[0][]);
    }

    /** select topN candidates closest to the base value */
    private double[][] selectCandidatesByBase(double[][] candidates, double[] base, int topN) {
        if (candidates == null || candidates.length == 0) {
            return candidates;
        }
        if (candidates.length <= topN) {
            return candidates;
        }
        Integer[] indices = new Integer[candidates.length];
        for (int j = 0; j < candidates.length; j++) {
            indices[j] = j;
        }
        Arrays.sort(indices, Comparator.comparingDouble(a -> distanceAbs(candidates[a], base)));
        double[][] selected = new double[topN][];
        for (int j = 0; j < topN; j++) {
            selected[j] = candidates[indices[j]];
        }
        return selected;
    }

    private double[] selectOptimalByReference(double[][] candidates, double[] reference) {
        double[] optimal = candidates[0];
        double minDis = Double.MAX_VALUE;
        for (double[] candidate : candidates) {
            double dis = delta(reference, candidate);
            if (dis < minDis) {
                minDis = dis;
                optimal = candidate;
            }
        }
        return optimal;
    }

    /**
     * Repair anomalies inside the initial window (no prediction available):
     * build ~3 interpolation candidates on normal points, then pick the best by domain-constraint consistency.
     */
    public void repairInitialWindow() {
        for (int i = 0; i <= initial_window[1]; i++) {
            if (td_anomalies[i]) {
                long repairStart = System.nanoTime();
                double[][] candidates = getInterpolationCandidates(i);
                double[] base = linearInterpolateFromNormals(i);
                // pick the candidate closest to domain constraints (via nearest neighbor of the base)
                double[] reference = kdTreeUtil.nearestNeighbor(base);
                double[] optimal_repair = candidates[0];
                double minDis = Double.MAX_VALUE;
                for (double[] candidate : candidates) {
                    double dis = distanceAbs(candidate, reference);
                    if (dis < minDis) {
                        minDis = dis;
                        optimal_repair = candidate;
                    }
                }
                if (shouldTimeIndex(i)) {
                    repair_time_ns += (System.nanoTime() - repairStart);
                }

                this.td_repaired[i] = optimal_repair;
                this.td_prediction[i] = base;
                regression_loss += delta(optimal_repair, base);
                prediction_regression_loss += delta(base, td[i]);
                n_repair_loss_count++;
                n_prediction_loss_count++;
            } else {
                this.td_prediction[i] = td_repaired[i].clone();
            }
        }
    }

    public void forwardRepairing(int p) {
        int i = initial_window[1] + 1;

        while (i < n) {
            if (td_anomalies[i] == Boolean.TRUE) {
                double[][] W_repaired = getWindow(this.td_repaired, i, p);

                // prediction phase
                long predStart = System.nanoTime();
                double[] x_repaired_predicted = arrayToList(prediction_model.predict(W_repaired));
                long predElapsed = System.nanoTime() - predStart;
                if (shouldTimeIndex(i)) {
                    prediction_time_ns += predElapsed;
                    prediction_call_count++;
                }

                // store predicted value
                this.td_prediction[i] = x_repaired_predicted.clone();

                // domain-constraint repair phase (candidate search + selection only)
                long repairStart = System.nanoTime();
                double[][] candidates =
                        this.kdTreeUtil.kNearestNeighbors(x_repaired_predicted, this.k);
                // when k > 3: use linear-interp base on nearby normals to keep top-3 candidates
                if (this.k > CANDIDATE_TOP_N) {
                    double[] base = linearInterpolateFromNormals(i);
                    candidates = selectCandidatesByBase(candidates, base, CANDIDATE_TOP_N);
                }
                double[] optimal_repair = selectOptimalByReference(candidates, x_repaired_predicted);
                if (shouldTimeIndex(i)) {
                    repair_time_ns += (System.nanoTime() - repairStart);
                }

                this.td_repaired[i] = optimal_repair;
                regression_loss += delta(optimal_repair, x_repaired_predicted);
                prediction_regression_loss += delta(x_repaired_predicted, td[i]);
                n_repair_loss_count++;
                n_prediction_loss_count++;
            } else {
                // normal point: keep current repaired value
                this.td_prediction[i] = td_repaired[i].clone();
            }
            i++;
        }
    }


    public void initList() {
        this.regression_loss = 0.0;
        this.prediction_regression_loss = 0.0;
        this.n_repair_loss_count = 0;
        this.n_prediction_loss_count = 0;
        this.actual_rounds = 0;
        td_prediction = new double[n][columnCnt];
        td_repaired = new double[n][columnCnt];
        this.model_train_time_ns = 0;
        this.prediction_time_ns = 0;
        this.repair_time_ns = 0;
        this.prediction_call_count = 0;
//        anomalies_in_repaired = new boolean[n];
    }

    private void deepCopyTdToRepaired() {
        for (int i = 0; i < n; i++) {
            td_repaired[i] = td[i].clone();
        }
    }

    /**
     * Multi-round repair: each round detects anomalies on the current series then repairs
     * from start to end, until no anomalies remain or the round limit t is reached.
     */
    public void repair() {
        call_std();
        findInitialWindow(p);
        initList();
        deepCopyTdToRepaired();
        // learn prediction model once from original normal points
        getOriginalAnomaliesAndLearnModel();

        for (int round = 1; round <= t; round++) {
            int anomalyCount;
            if (round == 1) {
                anomalyCount = countAnomalies();
            } else {
                anomalyCount = detectAnomaliesOnCurrent();
            }
            if (anomalyCount == 0) {
                System.out.println("ARMRepair stopped early at round " + round + ": no anomalies detected");
                break;
            }
            this.actual_rounds = round;
            System.out.println("ARMRepair round " + round + "/" + t + ": anomalies=" + anomalyCount);
            repairInitialWindow();
            forwardRepairing(p);
        }
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

    /** The model fitted during repair (same instance used in forward repair). */
    public TimeSeriesPredictor getPrediction_model() {
        return prediction_model;
    }

    public int getModelTrainEnd() {
        return modelTrainEnd;
    }

//    public boolean[] getAnomalies_in_repaired() {
//        return anomalies_in_repaired;
//    }

    public long getCost_time() {
        // for fair comparison / CSV: only domain-constraint repair phase
        return getRepair_time();
    }

    public long getWall_time() {
        return wall_time;
    }

    public int getActual_rounds() {
        return actual_rounds;
    }

    public int getT() {
        return t;
    }
    
    public long getModel_train_time() {
        return model_train_time_ns / 1_000_000L;
    }

    public long getPrediction_time() {
        return prediction_time_ns / 1_000_000L;
    }

    /** Average latency (ms) of one timed predict() call; 0 if none. */
    public double getAvg_prediction_time_ms() {
        if (prediction_call_count <= 0) {
            return 0.0;
        }
        return (prediction_time_ns / 1_000_000.0) / prediction_call_count;
    }

    public long getPrediction_call_count() {
        return prediction_call_count;
    }
    
    public long getRepair_time() {
        return repair_time_ns / 1_000_000L;
    }
    
    public long getTotal_repair_time() {
        return getPrediction_time() + getRepair_time();
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