import Algorithm.*;
import Algorithm.util.KDTreeUtil;
import Algorithm.util.DLinearUtil;
import Algorithm.util.PatchTSTUtil;
import Algorithm.util.TimeSeriesPredictor;
import Algorithm.util.VARUtil;
import Algorithm.util.MtcscUtil.Assist;
import Algorithm.util.MtcscUtil.TimeSeriesN;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;
import java.util.Scanner;

/**
 * Experiment entry point for ARM data repair evaluation.
 *
 * Sections:
 *   1. Config / dataset init
 *   2. Repair runners (ARM + baselines)
 *   3. I/O helpers
 *   4. main / session setup
 *   5. Forecast ablations (context / horizon, VAR & DLinear)
 *   6. Model comparison (VAR / DLinear / PatchTST)
 *   7. End-to-end repair benchmarks
 *   8. Sensitivity experiments (scale, error rate/range/length, params)
 */
public class Experiment {

    // -------------------------------------------------------------------------
    // 1. Config / dataset init
    // -------------------------------------------------------------------------

    static final String[] DATASETS = {"engine", "gps", "road", "weather", "traj"};

    static String DATA_BASE_PATH = "./data/";
    static String OUTPUT_BASE_PATH = "./model/data/";

    static String td_path;
    /** Path to domain-constraint CSV (files historically named master_data_*.csv). */
    static String md_path;
    static int td_len;
    static int md_len;

    static int error_rate;
    static double error_range;
    static int error_length;

    static double label_rate;  // IMR label ratio
    static int k;              // ARM: k-NN candidates
    static int p;              // ARM: autoregressive window
    static double eta;         // ARM: domain-constraint threshold
    static double beta;        // ARM initial E + AnomalyDetector residual threshold
    static int seed;

    /** Prefix length for CSDI vs ARM (diffusion is too slow on 100k+ points). */
    static final int CSDI_COMPARE_LEN = 4000;

    public static void init(int dataset_idx) {
        switch (dataset_idx) {
            case 0 -> {
                td_path = DATA_BASE_PATH + "engine/time_series_data_1596148.csv";
                md_path = DATA_BASE_PATH + "engine/master_data_14756.csv";
                td_len = 600 * 1000;
                md_len = 20000;
                error_rate = 5;
                error_range = 12.0;
                error_length = 5;
            }
            case 1 -> {
                td_path = DATA_BASE_PATH + "gps/time_series_data_1166375.csv";
                md_path = DATA_BASE_PATH + "gps/master_data_92110.csv";
                td_len = 200 * 1000;
                md_len = 80000;
                error_rate = 5;
                error_range = 12.0;
                error_length = 5;
            }
            case 2 -> {
                td_path = DATA_BASE_PATH + "road/time_series_data_1829660.csv";
                md_path = DATA_BASE_PATH + "road/master_data_66876.csv";
                td_len = 300 * 1000;
                md_len = 6000;
                error_rate = 5;
                error_range = 6.0;
                error_length = 5;
            }
            case 3 -> {
                td_path = DATA_BASE_PATH + "weather/time_series_data_390598.csv";
                md_path = DATA_BASE_PATH + "weather/master_data_524288.csv";
                td_len = 30 * 1000;
                md_len = 6000;
                error_rate = 5;
                error_range = 25.0;
                error_length = 5;
            }
            default -> {
                td_path = DATA_BASE_PATH + "traj/time_series_data_25168.csv";
                md_path = DATA_BASE_PATH + "traj/master_data_1180.csv";
                td_len = 25000;
                md_len = 1200;
                error_rate = 5;
                error_range = 6.0;
                error_length = 5;
            }
        }
        label_rate = 0.2;
        k = 7;
        p = 17;
        // eta = 0.7;
        eta = 1;
        // beta = 0.4;
        beta = 1;
        seed = 665;
    }

    /**
     * Loaded series + domain constraints for one run.
     * Domain-constraint CSVs remain named {@code master_data_*.csv} on disk.
     */
    static final class PreparedData {
        final long[] td_time;
        final String[] td_time_str;
        final double[][] td_clean;
        final double[][] td_dirty;
        final double[][] domainData;
        final double[][] td_label;
        final KDTreeUtil kdTree;
        final KDTreeUtil kdTreeComplete;
        final boolean[] detect_clean;
        final boolean[] td_bool;
        final boolean[] default_bool;

        PreparedData(long[] td_time, String[] td_time_str, double[][] td_clean, double[][] td_dirty,
                     double[][] domainData, double[][] td_label, KDTreeUtil kdTree,
                     KDTreeUtil kdTreeComplete, boolean[] detect_clean, boolean[] td_bool) {
            this.td_time = td_time;
            this.td_time_str = td_time_str;
            this.td_clean = td_clean;
            this.td_dirty = td_dirty;
            this.domainData = domainData;
            this.td_label = td_label;
            this.kdTree = kdTree;
            this.kdTreeComplete = kdTreeComplete;
            this.detect_clean = detect_clean;
            this.td_bool = td_bool;
            this.default_bool = new boolean[td_clean.length];
            Arrays.fill(this.default_bool, false);
        }
    }

    /** Load data, optionally inject noise, and build IMR labels. */
    static PreparedData prepareData(boolean injectNoise, boolean useRawAsDirty) throws Exception {
        LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
        long[] td_time = loadData.getTd_time();
        String[] td_time_str = loadData.getTd_time_str();
        double[][] td_clean = loadData.getTd_clean();
        double[][] domainData = loadData.getMd();
        KDTreeUtil kdTree = loadData.getKdTree();
        KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();

        double[][] td_dirty;
        if (useRawAsDirty) {
            td_dirty = loadData.getTd_raw_array();
        } else if (injectNoise) {
            td_dirty = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed)
                    .getTd_dirty();
        } else {
            td_dirty = td_clean;
        }

        boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);
        LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
        return new PreparedData(td_time, td_time_str, td_clean, td_dirty, domainData,
                labelData.getTd_label(), kdTree, kdTreeComplete, detect_clean, labelData.getTd_bool());
    }

    static void ensureDir(String path) {
        new File(path).mkdirs();
    }

    static void gcQuiet() {
        System.gc();
        Runtime.getRuntime().gc();
    }

    static void recordAnalysis(Analysis analysis) throws Exception {
        recordFile(analysis.getRMSE() + ",", "RMSE");
        recordFile(analysis.getPrecision() + ",", "Precision");
        recordFile(analysis.getRecall() + ",", "Recall");
        recordFile(analysis.getCost_time() + ",", "Time");
    }

    static void recordBlankLines(String suffix) throws Exception {
        recordFile(suffix, "RMSE");
        recordFile(suffix, "Precision");
        recordFile(suffix, "Recall");
        recordFile(suffix, "Time");
    }

    /**
     * Run ARM and baselines on prepared data.
     * @param methodCount 5 = ARM/ER/SCREEN/Lsgreedy/IMR; 6 adds MTCSC
     */
    static Analysis[] runRepairSuite(PreparedData d, int methodCount) throws Exception {
        Analysis[] results = new Analysis[methodCount];
        results[0] = armRepair(d.kdTree, d.td_time, d.td_clean, d.td_dirty, d.default_bool, d.detect_clean);
        if (methodCount > 1) {
            results[1] = erRepair(d.domainData, d.td_time, d.td_clean, d.td_dirty, d.default_bool, d.detect_clean);
        }
        if (methodCount > 2) {
            results[2] = screenRepair(d.td_time, d.td_clean, d.td_dirty, d.default_bool, d.detect_clean);
        }
        if (methodCount > 3) {
            results[3] = lsgreedyRepair(d.td_time, d.td_clean, d.td_dirty, d.default_bool, d.detect_clean);
        }
        if (methodCount > 4) {
            results[4] = imrRepair(d.td_time, d.td_clean, d.td_dirty, d.td_label, d.td_bool, d.detect_clean);
        }
        if (methodCount > 5) {
            int dim = d.td_clean[0].length;
            Assist assist = new Assist();
            TimeSeriesN timeSeriesN = new TimeSeriesN(d.td_time, d.td_dirty, d.td_clean);
            double S = assist.getSpeedN(timeSeriesN, 1.0, dim);
            results[5] = mtcscRepair(d.td_time, d.td_clean, d.td_dirty, d.default_bool, d.detect_clean, S, p, dim);
        }
        return results;
    }

    // -------------------------------------------------------------------------
    // 2. Repair runners (ARM + baselines)
    // -------------------------------------------------------------------------

    public static Analysis armRepair(KDTreeUtil kdTree, long[] td_time, double[][] td_clean,
                                     double[][] td_dirty, boolean[] td_bool, boolean[] detect_clean) {
        System.out.println("\nARMRepair");
        int columnCnt = td_clean[0].length;
        ARMRepair detector = new ARMRepair(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta);
        double[][] td_repair = detector.getTd_repaired();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, detector.getCost_time(),
                detect_clean, detect_repair);
    }

    public static Analysis armRepairDLinear(KDTreeUtil kdTree, long[] td_time, double[][] td_clean,
                                            double[][] td_dirty, boolean[] td_bool, boolean[] detect_clean) {
        System.out.println("\nARMRepair (DLinear)");
        int columnCnt = td_clean[0].length;
        DLinearUtil dlinearModel = new DLinearUtil(p, columnCnt);
        ARMRepair detector = new ARMRepair(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta, 1, dlinearModel);
        double[][] td_repair = detector.getTd_repaired();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta, dlinearModel);
        return new Analysis(td_time, td_clean, td_repair, td_bool, detector.getCost_time(),
                detect_clean, detect_repair);
    }

    public static Analysis domainNnRepair(KDTreeUtil kdTree, long[] td_time, double[][] td_clean,
                                          double[][] td_dirty, boolean[] td_bool, boolean[] detect_clean) {
        System.out.println("\nDomainNNRepair");
        int columnCnt = td_clean[0].length;
        DomainNNRepair detector = new DomainNNRepair(td_dirty, kdTree, td_time, columnCnt, eta);
        double[][] td_repair = detector.getTd_repaired();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, detector.getCost_time(),
                detect_clean, detect_repair);
    }

    public static Analysis modelOnlyRepair(long[] td_time, double[][] td_clean, double[][] td_dirty,
                                           boolean[] td_bool, boolean[] detect_clean) {
        System.out.println("\nModelOnlyRepair");
        int columnCnt = td_clean[0].length;
        ModelOnlyRepair detector = new ModelOnlyRepair(td_dirty, td_time, columnCnt, p, beta);
        double[][] td_repair = detector.getTd_repaired();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, detector.getCost_time(),
                detect_clean, detect_repair);
    }

    /**
     * CSDI trains on the dirty series, flags high reconstruction residual, then
     * imputes those points. Does not receive ARM's E mask.
     */
    public static Analysis csdiRepair(long[] td_time, double[][] td_clean, double[][] td_dirty,
                                      boolean[] td_bool, boolean[] detect_clean) {
        System.out.println("\nCSDIRepair");
        int columnCnt = td_clean[0].length;
        CSDIRepair detector = new CSDIRepair(td_dirty, td_time, columnCnt);
        double[][] td_repair = detector.getTd_repaired();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, detector.getCost_time(),
                detect_clean, detect_repair);
    }

    public static Analysis erRepair(double[][] md, long[] td_time, double[][] td_clean, double[][] td_dirty, boolean[] td_bool, boolean[] detect_clean) {
        System.out.println("\nERRepair");
        EditingRuleRepair editingRuleRepair = new EditingRuleRepair(td_time, td_dirty, md);
        double[][] td_repair = editingRuleRepair.getTd_repair();
        long cost_time = editingRuleRepair.getCost_time();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time, detect_clean, detect_repair);
    }

    public static Analysis screenRepair(long[] td_time, double[][] td_clean, double[][] td_dirty, boolean[] td_bool, boolean[] detect_clean) throws Exception {
        System.out.println("\nSCREEN");
        SCREEN screen = new SCREEN(td_time, td_dirty);
        double[][] td_repair = screen.getTd_repair();
        long cost_time = screen.getCost_time();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time, detect_clean, detect_repair);
    }

    public static Analysis lsgreedyRepair(long[] td_time, double[][] td_clean, double[][] td_dirty, boolean[] td_bool, boolean[] detect_clean) throws Exception {
        System.out.println("\nLsgreedy");
        Lsgreedy lsgreedy = new Lsgreedy(td_time, td_dirty);
        double[][] td_repair = lsgreedy.getTd_repair();
        long cost_time = lsgreedy.getCost_time();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time, detect_clean, detect_repair);
    }

    public static Analysis imrRepair(long[] td_time, double[][] td_clean, double[][] td_dirty, double[][] td_label, boolean[] td_bool, boolean[] detect_clean) {
        System.out.println("\nIMR");
        IMR imr = new IMR(td_time, td_dirty, td_label, td_bool);
        double[][] td_repair = imr.getTd_repair();
        long cost_time = imr.getCost_time();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time, detect_clean, detect_repair);
    }

    public static Analysis mtcscRepair(long[] td_time, double[][] td_clean, double[][] td_dirty, boolean[] td_bool, boolean[] detect_clean, double S, int T, int dim) throws Exception {
        System.out.println("\nMTCSC");
        MTCSC mtcsc = new MTCSC(td_time, td_dirty, td_clean, S, T, dim);
        double[][] td_repair = mtcsc.getTd_repair();
        long cost_time = mtcsc.getCost_time();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time, detect_clean, detect_repair);
    }


    // -------------------------------------------------------------------------
    // 3. I/O helpers
    // -------------------------------------------------------------------------

    public static void writeDataToCSV(String[] td_time, double[][] data, String csvFile) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(csvFile))) {
//            bw.write(header);
//            bw.newLine();
            for (int i = 0; i < data.length; i++) {
                double[] row = data[i];
                String time = td_time[i];
                StringBuilder line = new StringBuilder();
//                System.out.println(time);
                line.append(time);
                for (double value : row) {
                    if (line.length() > 0) {
                        line.append(",");
                    }
                    line.append(value);
                }
                bw.write(line.toString());
                bw.newLine();
            }
        } catch (IOException e) {
            System.err.println("ERROR: " + e.getMessage());
        }
    }

    /**
     * Write timestamp-aligned compare CSV for test segment [startIdx, n).
     * Per dimension: clean, dirty (only if != clean), pred_dirty, repaired (only if != dirty), pred_repaired.
     */
    public static void writeForecastCompareCSV(String[] td_time, double[][] clean, double[][] dirty,
                                               double[][] repaired, double[][] predDirty,
                                               double[][] predRepaired, int startIdx, String csvFile) {
        int len = predDirty.length;
        int cols = predDirty[0].length;
        if (predRepaired.length != len
                || clean.length < startIdx + len
                || dirty.length < startIdx + len
                || repaired.length < startIdx + len) {
            throw new IllegalArgumentException("Series length mismatch for forecast compare CSV");
        }
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(csvFile))) {
            StringBuilder header = new StringBuilder("timestamp");
            for (int c = 0; c < cols; c++) {
                String s = cols == 1 ? "" : ("_" + c);
                header.append(",clean").append(s)
                        .append(",dirty").append(s)
                        .append(",pred_dirty").append(s)
                        .append(",repaired").append(s)
                        .append(",pred_repaired").append(s);
            }
            bw.write(header.toString());
            bw.newLine();
            for (int i = 0; i < len; i++) {
                int idx = startIdx + i;
                StringBuilder line = new StringBuilder(td_time[idx]);
                for (int c = 0; c < cols; c++) {
                    double cleanV = clean[idx][c];
                    double dirtyV = dirty[idx][c];
                    double repairedV = repaired[idx][c];
                    line.append(',').append(cleanV);
                    line.append(',');
                    if (Double.compare(dirtyV, cleanV) != 0) {
                        line.append(dirtyV);
                    }
                    line.append(',').append(predDirty[i][c]);
                    line.append(',');
                    if (Double.compare(repairedV, dirtyV) != 0) {
                        line.append(repairedV);
                    }
                    line.append(',').append(predRepaired[i][c]);
                }
                bw.write(line.toString());
                bw.newLine();
            }
        } catch (IOException e) {
            System.err.println("ERROR: " + e.getMessage());
        }
    }

    public static void recordFile(String string, String type) throws Exception {
        File resultsDir = new File("./results");
        if (!resultsDir.exists()) {
            resultsDir.mkdirs(); 
        }
        FileWriter fileWritter = new FileWriter("./results/exp" + type + ".txt", true);
        BufferedWriter bw = new BufferedWriter(fileWritter);
        bw.write(string);
        bw.close();
    }

    // -------------------------------------------------------------------------
    // 4. main / session setup
    // -------------------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        starts();

        // --- enable one experiment at a time ---
        // varyingModel(5);
        // varyingModelDLinear(5);
        // varyingModelPatchTST(5);
        // varyingForecastContext(5, 2);
        // varyingForecastHorizon(5, 2);
        // varyingForecastContextDLinear(5, 2);
        // varyingForecastHorizonDLinear(5, 2);
        // get_arm_repaired(5);
        // get_data_repaired(5);
        // get_normalized_repaired(5);
        // get_baseline_repaired(5);
        get_detection_pr();
        // get_csdi_repaired(5);
        // get_domain_noise_arm(5);
        // main_error_rate();
        // get_prefix_error_arm();
        // main_error_range();
        // main_error_length();
        // main_parameters(false, true, false, false);
    }

    public static void starts() throws Exception {
        SimpleDateFormat df = new SimpleDateFormat("yyyy_MM_dd_HH_MM_SS");
        String current_time = df.format(new Date());
        System.out.println("current time " + current_time);
        recordFile("\n" + current_time + "\n", "RMSE");
        recordFile("\n" + current_time + "\n", "Precision");
        recordFile("\n" + current_time + "\n", "Recall");
        recordFile("\n" + current_time + "\n", "Time");
        recordFile("\n" + current_time + "\n", "RegressionLoss");
    }
    
    /**
     * Clear result files (optional; use when starting a new experiment).
     * Note: this removes all previous results.
     */
    public static void clearResults() throws Exception {
        String[] types = {"RMSE", "Precision", "Recall", "Time", "RegressionLoss"};
        for (String type : types) {
            File resultFile = new File("./results/exp" + type + ".txt");
            if (resultFile.exists()) {
                try (FileWriter fw = new FileWriter(resultFile, false)) {
                    fw.write("");  // clear content (false = overwrite mode)
                }
                System.out.println("Cleared results file: exp" + type + ".txt");
            }
        }
    }

    // -------------------------------------------------------------------------
    // 5–6. Model comparison & forecast ablations
    // -------------------------------------------------------------------------

    public static void varyingModel(int error_rate) throws Exception {
        recordFile("\nERROR RATE " + error_rate + " (ARM+VAR forecast)\n", "RMSE");
        recordFile("\nERROR RATE " + error_rate + " (ARM+VAR forecast) Time=train,repair,pred\n", "Time");
        // recordFile("\nERROR RATE " + error_rate + " (ARM+VAR forecast)\n", "Precision");
        // recordFile("\nERROR RATE " + error_rate + " (ARM+VAR forecast)\n", "Recall");
        // recordFile("\nERROR RATE " + error_rate + " (ARM+VAR forecast)\n", "RegressionLoss");
        String data_csv_path = OUTPUT_BASE_PATH;
        String[] datasets = DATASETS;
        final double trainRatio = 0.8;

        for (int dataset_idx = 0; dataset_idx < 5; dataset_idx++) {
            String file_path = data_csv_path + datasets[dataset_idx] + "/";
            File directory = new File(file_path);
            if (!directory.exists()) {
                directory.mkdirs();
            }

            init(dataset_idx);

            LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
            long[] td_time = loadData.getTd_time();
            String[] td_time_str = loadData.getTd_time_str();
            double[][] td_clean = loadData.getTd_clean();
            KDTreeUtil kdTree = loadData.getKdTree();
            KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
            writeDataToCSV(td_time_str, td_clean, file_path + "clean_t.csv");

            AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
            double[][] td_dirty = addNoise.getTd_dirty();
            writeDataToCSV(td_time_str, td_dirty, file_path + "dirty_" + error_rate + "_t.csv");

            int n = td_dirty.length;
            int columnCnt = td_clean[0].length;
            int trainLen = (int) (n * trainRatio);
            if (trainLen <= p || trainLen >= n) {
                throw new IllegalStateException("Invalid 80/20 split for " + datasets[dataset_idx]);
            }

            recordFile("\n" + datasets[dataset_idx] + ",", "RMSE");
            // recordFile("\n" + datasets[dataset_idx] + ",", "Precision");
            // recordFile("\n" + datasets[dataset_idx] + ",", "Recall");
            recordFile("\n" + datasets[dataset_idx] + ",", "Time");
            // recordFile("\n" + datasets[dataset_idx] + ",", "RegressionLoss");

            System.out.println("\n===== ARM+VAR forecast eval: " + datasets[dataset_idx] + " =====");
            System.out.println("n=" + n + ", trainLen(80%)=" + trainLen + ", testLen(20%)=" + (n - trainLen));

            // ARM repair: model trained only on conflict-free points in first 80%
            VARUtil varModel = new VARUtil(p);
            ARMRepair armRepair = new ARMRepair(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta, 3, varModel, trainLen);
            double[][] td_repaired = armRepair.getTd_repaired();
            TimeSeriesPredictor model = armRepair.getPrediction_model(); // same model, no retrain
            writeDataToCSV(td_time_str, td_repaired, file_path + "ARM_VAR_repaired.csv");

            // Test on last 20%: (1) predict on dirty  (2) predict on repaired — same model
            long predStart = System.nanoTime();
            double[][] predDirty = rollingPredictFrom(model, td_dirty, trainLen, p);
            long predMs = (System.nanoTime() - predStart) / 1_000_000L;
            double rmseDirty = rmseAgainstClean(predDirty, td_clean, trainLen);
            double[][] predRepaired = rollingPredictFrom(model, td_repaired, trainLen, p);
            double rmseRepaired = rmseAgainstClean(predRepaired, td_clean, trainLen);

            long trainMs = armRepair.getModel_train_time();
            long repairMs = armRepair.getRepair_time(); // test-only when modelTrainEnd < n

            writeForecastCompareCSV(td_time_str, td_clean, td_dirty, td_repaired,
                    predDirty, predRepaired, trainLen, file_path + "ARM_VAR_forecast_compare.csv");

            System.out.println("Results for " + datasets[dataset_idx] + ":");
            System.out.println("  RMSE (model on dirty test):     " + String.format("%.3f", rmseDirty));
            System.out.println("  RMSE (model on repaired test):  " + String.format("%.3f", rmseRepaired));
            System.out.println("  Time (ms): train=" + trainMs
                    + ", repair=" + repairMs
                    + ", pred=" + predMs
                    + "; ARM rounds: " + armRepair.getActual_rounds() + "/" + armRepair.getT());

            recordFile(String.format("%.3f", rmseDirty) + ",", "RMSE");
            recordFile(String.format("%.3f", rmseRepaired) + ",", "RMSE");
            // Time columns: train | repair (test) | pred (test)
            recordFile(trainMs + ",", "Time");
            recordFile(repairMs + ",", "Time");
            recordFile(predMs + ",", "Time");
            // recordFile("NA,", "Precision");
            // recordFile("NA,", "Precision");
            // recordFile("NA,", "Recall");
            // recordFile("NA,", "Recall");
            // recordFile("NA,", "RegressionLoss");
            // recordFile("NA,", "RegressionLoss");

            System.gc();
            Runtime.getRuntime().gc();
        }
    }

    /**
     * Ablation: ARM repair still uses p→1; forecast eval varies context length.
     * Context lengths: local vector from 5 to 20 (step 1). Each still predicts 1 step.
     * Saves dirty/repaired RMSE per context length.
     */
    public static void varyingForecastContext(int error_rate, int dataset_idx) throws Exception {
        String[] datasets = DATASETS;
        if (dataset_idx < 0 || dataset_idx >= datasets.length) {
            throw new IllegalArgumentException("dataset_idx must be in [0, " + (datasets.length - 1) + "]");
        }
        final double trainRatio = 0.8;
        // local vector: context window lengths to sweep
        int[] contextLens = new int[16]; // 5..20
        for (int i = 0; i < contextLens.length; i++) {
            contextLens[i] = 5 + i;
        }

        String dataset = datasets[dataset_idx];
        String file_path = OUTPUT_BASE_PATH + dataset + "/";
        new File(file_path).mkdirs();
        String outDir = "./results/forecast_context/" + dataset;
        new File(outDir).mkdirs();

        init(dataset_idx);
        LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
        long[] td_time = loadData.getTd_time();
        String[] td_time_str = loadData.getTd_time_str();
        double[][] td_clean = loadData.getTd_clean();
        KDTreeUtil kdTree = loadData.getKdTree();
        KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();

        AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
        double[][] td_dirty = addNoise.getTd_dirty();

        int n = td_dirty.length;
        int columnCnt = td_clean[0].length;
        int trainLen = (int) (n * trainRatio);
        int maxCtx = contextLens[contextLens.length - 1];
        if (trainLen <= maxCtx || trainLen >= n) {
            throw new IllegalStateException("Invalid split for context sweep on " + dataset);
        }

        System.out.println("\n===== Forecast context sweep: " + dataset + " er=" + error_rate + " =====");
        System.out.println("Repair window p=" + p + "→1; eval contexts=" + Arrays.toString(contextLens));

        // repair once with fixed p
        VARUtil repairModel = new VARUtil(p);
        ARMRepair armRepair = new ARMRepair(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta, 3, repairModel, trainLen);
        double[][] td_repaired = armRepair.getTd_repaired();
        writeDataToCSV(td_time_str, td_repaired, file_path + "ARM_VAR_repaired_ctx_exp.csv");
        long repairMs = armRepair.getRepair_time(); 
        ArrayList<ArrayList<Double>> trainSamples =
                collectConflictFreeSamples(td_dirty, kdTree, eta, trainLen);

        String csvPath = outDir + "/rmse_er" + error_rate + ".csv";
        String timePath = outDir + "/time_er" + error_rate + ".csv";
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(csvPath));
             BufferedWriter tw = new BufferedWriter(new FileWriter(timePath))) {
            bw.write("context_len,rmse_dirty,rmse_repaired");
            bw.newLine();
            tw.write("context_len,train_ms,repair_ms,pred_ms");
            tw.newLine();

            for (int ctx : contextLens) {
                if (trainSamples.size() <= ctx) {
                    System.out.println("  skip ctx=" + ctx + " (not enough train samples)");
                    continue;
                }
                VARUtil evalModel = new VARUtil(ctx);
                long fitStart = System.nanoTime();
                evalModel.fit(trainSamples);
                long trainMs = (System.nanoTime() - fitStart) / 1_000_000L;

                long predStart = System.nanoTime();
                double[][] predDirty = rollingPredictFrom(evalModel, td_dirty, trainLen, ctx);
                long predMs = (System.nanoTime() - predStart) / 1_000_000L;
                double[][] predRepaired = rollingPredictFrom(evalModel, td_repaired, trainLen, ctx);
                double rmseDirty = rmseAgainstClean(predDirty, td_clean, trainLen);
                double rmseRepaired = rmseAgainstClean(predRepaired, td_clean, trainLen);

                bw.write(ctx + "," + String.format("%.6f", rmseDirty) + ","
                        + String.format("%.6f", rmseRepaired));
                bw.newLine();
                tw.write(ctx + "," + trainMs + "," + repairMs + "," + predMs);
                tw.newLine();

                System.out.println("  ctx=" + ctx
                        + "  dirty=" + String.format("%.3f", rmseDirty)
                        + "  repaired=" + String.format("%.3f", rmseRepaired)
                        + "  time(ms): train=" + trainMs
                        + ", repair=" + repairMs
                        + ", pred=" + predMs);
            }
        }
        System.out.println("Wrote " + csvPath);
        System.out.println("Wrote " + timePath);
    }

    /**
     * Ablation: ARM repair still uses p→1; forecast eval varies prediction horizon 1..10.
     * Context window fixed at p (17). Multi-step via recursive one-step predict.
     */
    public static void varyingForecastHorizon(int error_rate, int dataset_idx) throws Exception {
        String[] datasets = DATASETS;
        if (dataset_idx < 0 || dataset_idx >= datasets.length) {
            throw new IllegalArgumentException("dataset_idx must be in [0, " + (datasets.length - 1) + "]");
        }
        final double trainRatio = 0.8;
        final int maxHorizon = 20;
        final int p = 17;

        String dataset = datasets[dataset_idx];
        String file_path = OUTPUT_BASE_PATH + dataset + "/";
        new File(file_path).mkdirs();
        String outDir = "./results/forecast_horizon/" + dataset;
        new File(outDir).mkdirs();

        init(dataset_idx);
        LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
        long[] td_time = loadData.getTd_time();
        String[] td_time_str = loadData.getTd_time_str();
        double[][] td_clean = loadData.getTd_clean();
        KDTreeUtil kdTree = loadData.getKdTree();
        KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();

        AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
        double[][] td_dirty = addNoise.getTd_dirty();

        int n = td_dirty.length;
        int columnCnt = td_clean[0].length;
        int trainLen = (int) (n * trainRatio);
        if (trainLen <= p || trainLen + maxHorizon > n) {
            throw new IllegalStateException("Invalid split for horizon sweep on " + dataset);
        }

        System.out.println("\n===== Forecast horizon sweep: " + dataset + " er=" + error_rate + " =====");
        System.out.println("Context fixed p=" + p + "; horizon=1.." + maxHorizon + " (direct multi-output)");

        // ARM repair still p→1
        VARUtil repairModel = new VARUtil(p);
        ARMRepair armRepair = new ARMRepair(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta, 3, repairModel, trainLen);
        double[][] td_repaired = armRepair.getTd_repaired();
        writeDataToCSV(td_time_str, td_repaired, file_path + "ARM_VAR_repaired_horizon_exp.csv");
        long repairMs = armRepair.getRepair_time();

        ArrayList<ArrayList<Double>> trainSamples =
                collectConflictFreeSamples(td_dirty, kdTree, eta, trainLen);

        String csvPath = outDir + "/rmse_er" + error_rate + ".csv";
        String timePath = outDir + "/time_er" + error_rate + ".csv";
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(csvPath));
             BufferedWriter tw = new BufferedWriter(new FileWriter(timePath))) {
            bw.write("horizon,rmse_dirty,rmse_repaired");
            bw.newLine();
            tw.write("horizon,train_ms,repair_ms,pred_ms");
            tw.newLine();

            for (int h = 1; h <= maxHorizon; h++) {
                // re-fit direct multi-output VAR: p → h in one shot
                VARUtil evalModel = new VARUtil(p);
                long fitStart = System.nanoTime();
                evalModel.fit(trainSamples, h);
                long trainMs = (System.nanoTime() - fitStart) / 1_000_000L;

                String valueDir = "./results/forecast_horizon/value";
                new File(valueDir).mkdirs();
                String valueCsv = valueDir + "/horizon_" + h + ".csv";

                long[] predStats = new long[2];
                double[] rmses = evalHorizonAndWriteValues(evalModel, td_dirty, td_repaired, td_clean,
                        td_time_str, trainLen, p, h, predStats, valueCsv);
                double rmseDirty = rmses[0];
                double rmseRepaired = rmses[1];
                long predMs = predStats[0] / 1_000_000L;

                bw.write(h + "," + String.format("%.6f", rmseDirty) + ","
                        + String.format("%.6f", rmseRepaired));
                bw.newLine();
                tw.write(h + "," + trainMs + "," + repairMs + "," + predMs);
                tw.newLine();

                System.out.println("  h=" + h
                        + "  dirty=" + String.format("%.3f", rmseDirty)
                        + "  repaired=" + String.format("%.3f", rmseRepaired)
                        + "  time(ms): train=" + trainMs
                        + ", repair=" + repairMs
                        + ", pred=" + predMs
                        + " (predict calls=" + predStats[1] + ")"
                        + "  values → " + valueCsv);
            }
        }
        System.out.println("Wrote " + csvPath);
        System.out.println("Wrote " + timePath);
    }

    /**
     * Same as {@link #varyingForecastContext} but ARM repair + eval models use DLinear.
     * For each context length, a DLinear with seq_len=ctx is re-fit for forecast eval.
     */
    public static void varyingForecastContextDLinear(int error_rate, int dataset_idx) throws Exception {
        String[] datasets = DATASETS;
        if (dataset_idx < 0 || dataset_idx >= datasets.length) {
            throw new IllegalArgumentException("dataset_idx must be in [0, " + (datasets.length - 1) + "]");
        }
        final double trainRatio = 0.8;
        int[] contextLens = new int[16]; // 5..20
        for (int i = 0; i < contextLens.length; i++) {
            contextLens[i] = 5 + i;
        }

        String dataset = datasets[dataset_idx];
        String file_path = OUTPUT_BASE_PATH + dataset + "/";
        new File(file_path).mkdirs();
        String outDir = "./results/forecast_context/dlinear/" + dataset;
        new File(outDir).mkdirs();

        init(dataset_idx);
        LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
        long[] td_time = loadData.getTd_time();
        String[] td_time_str = loadData.getTd_time_str();
        double[][] td_clean = loadData.getTd_clean();
        KDTreeUtil kdTree = loadData.getKdTree();
        KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();

        AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
        double[][] td_dirty = addNoise.getTd_dirty();

        int n = td_dirty.length;
        int columnCnt = td_clean[0].length;
        int trainLen = (int) (n * trainRatio);
        int maxCtx = contextLens[contextLens.length - 1];
        if (trainLen <= maxCtx || trainLen >= n) {
            throw new IllegalStateException("Invalid split for context sweep on " + dataset);
        }

        System.out.println("\n===== Forecast context sweep (DLinear): " + dataset + " er=" + error_rate + " =====");
        System.out.println("Repair window p=" + p + "→1; eval contexts=" + Arrays.toString(contextLens));

        DLinearUtil repairModel = new DLinearUtil(p, columnCnt, "./python/dlinear_model.py",
                "./models/forecast_context_repair_" + dataset, true, 100, 0.001);
        ARMRepair armRepair = new ARMRepair(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta, 1, repairModel, trainLen);
        double[][] td_repaired = armRepair.getTd_repaired();
        writeDataToCSV(td_time_str, td_repaired, file_path + "ARM_DLinear_repaired_ctx_exp.csv");
        long repairMs = armRepair.getRepair_time();
        repairModel.close();

        ArrayList<ArrayList<Double>> trainSamples =
                collectConflictFreeSamples(td_dirty, kdTree, eta, trainLen);

        String csvPath = outDir + "/rmse_er" + error_rate + ".csv";
        String timePath = outDir + "/time_er" + error_rate + ".csv";
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(csvPath));
             BufferedWriter tw = new BufferedWriter(new FileWriter(timePath))) {
            bw.write("context_len,rmse_dirty,rmse_repaired");
            bw.newLine();
            tw.write("context_len,train_ms,repair_ms,pred_ms");
            tw.newLine();

            for (int ctx : contextLens) {
                if (trainSamples.size() <= ctx) {
                    System.out.println("  skip ctx=" + ctx + " (not enough train samples)");
                    continue;
                }
                DLinearUtil evalModel = new DLinearUtil(ctx, columnCnt, "./python/dlinear_model.py",
                        "./models/forecast_context_eval_" + dataset + "_ctx" + ctx, true, 100, 0.001);
                long fitStart = System.nanoTime();
                evalModel.fit(trainSamples);
                long trainMs = (System.nanoTime() - fitStart) / 1_000_000L;

                long predStart = System.nanoTime();
                double[][] predDirty = rollingPredictFrom(evalModel, td_dirty, trainLen, ctx);
                long predMs = (System.nanoTime() - predStart) / 1_000_000L;
                double[][] predRepaired = rollingPredictFrom(evalModel, td_repaired, trainLen, ctx);
                double rmseDirty = rmseAgainstClean(predDirty, td_clean, trainLen);
                double rmseRepaired = rmseAgainstClean(predRepaired, td_clean, trainLen);
                evalModel.close();

                bw.write(ctx + "," + String.format("%.6f", rmseDirty) + ","
                        + String.format("%.6f", rmseRepaired));
                bw.newLine();
                tw.write(ctx + "," + trainMs + "," + repairMs + "," + predMs);
                tw.newLine();

                System.out.println("  ctx=" + ctx
                        + "  dirty=" + String.format("%.3f", rmseDirty)
                        + "  repaired=" + String.format("%.3f", rmseRepaired)
                        + "  time(ms): train=" + trainMs
                        + ", repair=" + repairMs
                        + ", pred=" + predMs);
            }
        }
        System.out.println("Wrote " + csvPath);
        System.out.println("Wrote " + timePath);
    }

    /**
     * Same as {@link #varyingForecastHorizon} but ARM repair + forecast use DLinear.
     * Context fixed at p; horizon 1..10; no retrain between horizons.
     */
    public static void varyingForecastHorizonDLinear(int error_rate, int dataset_idx) throws Exception {
        String[] datasets = DATASETS;
        if (dataset_idx < 0 || dataset_idx >= datasets.length) {
            throw new IllegalArgumentException("dataset_idx must be in [0, " + (datasets.length - 1) + "]");
        }
        final double trainRatio = 0.8;
        final int maxHorizon = 10;

        String dataset = datasets[dataset_idx];
        String file_path = OUTPUT_BASE_PATH + dataset + "/";
        new File(file_path).mkdirs();
        String outDir = "./results/forecast_horizon/dlinear/" + dataset;
        new File(outDir).mkdirs();

        init(dataset_idx);
        LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
        long[] td_time = loadData.getTd_time();
        String[] td_time_str = loadData.getTd_time_str();
        double[][] td_clean = loadData.getTd_clean();
        KDTreeUtil kdTree = loadData.getKdTree();
        KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();

        AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
        double[][] td_dirty = addNoise.getTd_dirty();

        int n = td_dirty.length;
        int columnCnt = td_clean[0].length;
        int trainLen = (int) (n * trainRatio);
        if (trainLen <= p || trainLen + maxHorizon > n) {
            throw new IllegalStateException("Invalid split for horizon sweep on " + dataset);
        }

        System.out.println("\n===== Forecast horizon sweep (DLinear): " + dataset + " er=" + error_rate + " =====");
        System.out.println("Context fixed p=" + p + "; horizon=1.." + maxHorizon + " (direct multi-output)");

        DLinearUtil repairModel = new DLinearUtil(p, columnCnt, 1, "./python/dlinear_model.py",
                "./models/forecast_horizon_repair_" + dataset, true, 100, 0.001);
        ARMRepair armRepair = new ARMRepair(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta, 1, repairModel, trainLen);
        double[][] td_repaired = armRepair.getTd_repaired();
        writeDataToCSV(td_time_str, td_repaired, file_path + "ARM_DLinear_repaired_horizon_exp.csv");
        long repairMs = armRepair.getRepair_time();
        repairModel.close();

        ArrayList<ArrayList<Double>> trainSamples =
                collectConflictFreeSamples(td_dirty, kdTree, eta, trainLen);

        String csvPath = outDir + "/rmse_er" + error_rate + ".csv";
        String timePath = outDir + "/time_er" + error_rate + ".csv";
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(csvPath));
             BufferedWriter tw = new BufferedWriter(new FileWriter(timePath))) {
            bw.write("horizon,rmse_dirty,rmse_repaired");
            bw.newLine();
            tw.write("horizon,train_ms,repair_ms,pred_ms");
            tw.newLine();

            for (int h = 1; h <= maxHorizon; h++) {
                DLinearUtil evalModel = new DLinearUtil(p, columnCnt, h, "./python/dlinear_model.py",
                        "./models/forecast_horizon_eval_" + dataset + "_h" + h, true, 100, 0.001);
                long fitStart = System.nanoTime();
                evalModel.fit(trainSamples);
                long trainMs = (System.nanoTime() - fitStart) / 1_000_000L;

                long[] predStats = new long[2];
                double rmseDirty = rmseRollingMultiStep(evalModel, td_dirty, td_clean, trainLen, p, h, predStats);
                double rmseRepaired = rmseRollingMultiStep(evalModel, td_repaired, td_clean, trainLen, p, h, null);
                long predMs = predStats[0] / 1_000_000L;
                evalModel.close();

                bw.write(h + "," + String.format("%.6f", rmseDirty) + ","
                        + String.format("%.6f", rmseRepaired));
                bw.newLine();
                tw.write(h + "," + trainMs + "," + repairMs + "," + predMs);
                tw.newLine();

                System.out.println("  h=" + h
                        + "  dirty=" + String.format("%.3f", rmseDirty)
                        + "  repaired=" + String.format("%.3f", rmseRepaired)
                        + "  time(ms): train=" + trainMs
                        + ", repair=" + repairMs
                        + ", pred=" + predMs
                        + " (predict calls=" + predStats[1] + ")");
            }
        }
        System.out.println("Wrote " + csvPath);
        System.out.println("Wrote " + timePath);
    }

    /** Conflict-free samples in [0, endExclusive), same rule as ARMRepair training. */
    private static ArrayList<ArrayList<Double>> collectConflictFreeSamples(
            double[][] td, KDTreeUtil kdTree, double eta, int endExclusive) {
        int n = Math.min(td.length, endExclusive);
        int cols = td[0].length;
        double[] std = computeStd(td);
        ArrayList<ArrayList<Double>> samples = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            double[] nn = kdTree.nearestNeighbor(td[i]);
            if (!(signedDelta(td[i], nn, std) > eta)) {
                ArrayList<Double> sample = new ArrayList<>(cols);
                for (int c = 0; c < cols; c++) {
                    sample.add(td[i][c]);
                }
                samples.add(sample);
            }
        }
        return samples;
    }

    /**
     * Same sliding / direct multi-step eval as {@link #rmseRollingMultiStep}, but also writes every
     * predicted point to {@code valueCsvPath}.
     * Returns {@code [rmse_dirty, rmse_repaired]}.
     * CSV columns: block,step,timestamp,clean_*,pred_dirty_*,pred_repaired_*
     */
    private static double[] evalHorizonAndWriteValues(TimeSeriesPredictor model,
                                                      double[][] dirty, double[][] repaired,
                                                      double[][] clean, String[] td_time_str,
                                                      int startIdx, int windowSize, int horizon,
                                                      long[] predStats, String valueCsvPath)
            throws IOException {
        if (startIdx < windowSize) {
            throw new IllegalArgumentException("startIdx must be >= windowSize");
        }
        int n = dirty.length;
        int cols = dirty[0].length;
        int stride = horizon;
        double sseDirty = 0.0;
        double sseRepaired = 0.0;
        long nPoints = 0;
        long predNs = 0;
        long predCalls = 0;
        int block = 0;

        File valueParent = new File(valueCsvPath).getParentFile();
        if (valueParent != null) {
            valueParent.mkdirs();
        }

        try (BufferedWriter vw = new BufferedWriter(new FileWriter(valueCsvPath))) {
            StringBuilder header = new StringBuilder("block,step,timestamp");
            for (int c = 0; c < cols; c++) {
                header.append(",clean_").append(c);
            }
            for (int c = 0; c < cols; c++) {
                header.append(",pred_dirty_").append(c);
            }
            for (int c = 0; c < cols; c++) {
                header.append(",pred_repaired_").append(c);
            }
            vw.write(header.toString());
            vw.newLine();

            for (int winStart = startIdx - windowSize; winStart + windowSize + horizon - 1 < n; winStart += stride) {
                double[][] windowDirty = new double[windowSize][cols];
                double[][] windowRepaired = new double[windowSize][cols];
                for (int j = 0; j < windowSize; j++) {
                    windowDirty[j] = dirty[winStart + j].clone();
                    windowRepaired[j] = repaired[winStart + j].clone();
                }

                long t0 = System.nanoTime();
                double[][] predDirty = model.predictHorizon(windowDirty, horizon);
                predNs += System.nanoTime() - t0;
                predCalls++;
                double[][] predRepaired = model.predictHorizon(windowRepaired, horizon);

                for (int step = 0; step < horizon; step++) {
                    int targetIdx = winStart + windowSize + step;
                    StringBuilder line = new StringBuilder();
                    line.append(block).append(',').append(step).append(',')
                            .append(td_time_str != null ? td_time_str[targetIdx] : targetIdx);
                    for (int c = 0; c < cols; c++) {
                        line.append(',').append(clean[targetIdx][c]);
                    }
                    for (int c = 0; c < cols; c++) {
                        line.append(',').append(predDirty[step][c]);
                        double diff = clean[targetIdx][c] - predDirty[step][c];
                        sseDirty += diff * diff;
                    }
                    for (int c = 0; c < cols; c++) {
                        line.append(',').append(predRepaired[step][c]);
                        double diff = clean[targetIdx][c] - predRepaired[step][c];
                        sseRepaired += diff * diff;
                    }
                    vw.write(line.toString());
                    vw.newLine();
                    nPoints++;
                }
                block++;
            }
        }

        if (predStats != null && predStats.length >= 2) {
            predStats[0] = predNs;
            predStats[1] = predCalls;
        }
        if (nPoints == 0) {
            return new double[]{Double.NaN, Double.NaN};
        }
        return new double[]{
                Math.sqrt(sseDirty / nPoints / cols),
                Math.sqrt(sseRepaired / nPoints / cols)
        };
    }

    /**
     * Block multi-step forecast with stride = horizon:
     * use window of length p to directly predict the next h points in one shot,
     * then slide forward by h so prediction blocks are contiguous and cover the test segment.
     * RMSE is over all predicted points vs clean (same formula as {@link #rmseAgainstClean}).
     */
    private static double rmseRollingMultiStep(TimeSeriesPredictor model, double[][] context,
                                              double[][] clean, int startIdx, int windowSize,
                                              int horizon, long[] predStats) {
        if (startIdx < windowSize) {
            throw new IllegalArgumentException("startIdx must be >= windowSize");
        }
        if (horizon < 1) {
            throw new IllegalArgumentException("horizon must be >= 1");
        }
        int n = context.length;
        int cols = context[0].length;
        if (startIdx + horizon > n) {
            throw new IllegalArgumentException("Not enough room for horizon");
        }

        // slide by h: prediction blocks [startIdx, startIdx+h), [startIdx+h, startIdx+2h), ...
        // cover the test segment without gaps
        int stride = horizon;
        double sse = 0.0;
        long nPoints = 0;
        long predNs = 0;
        long predCalls = 0;

        for (int winStart = startIdx - windowSize; winStart + windowSize + horizon - 1 < n; winStart += stride) {
            double[][] window = new double[windowSize][cols];
            for (int j = 0; j < windowSize; j++) {
                window[j] = context[winStart + j].clone();
            }
            // one-shot direct multi-step prediction of h points
            long t0 = System.nanoTime();
            double[][] preds = model.predictHorizon(window, horizon);
            predNs += System.nanoTime() - t0;
            predCalls++;

            for (int step = 0; step < horizon; step++) {
                int targetIdx = winStart + windowSize + step;
                for (int c = 0; c < cols; c++) {
                    double diff = clean[targetIdx][c] - preds[step][c];
                    sse += diff * diff;
                }
                nPoints++;
            }
        }

        if (predStats != null && predStats.length >= 2) {
            predStats[0] = predNs;
            predStats[1] = predCalls;
        }
        if (nPoints == 0) {
            return Double.NaN;
        }
        // same as rmseAgainstClean: sqrt(sse / N / D) over all predicted points
        return Math.sqrt(sse / nPoints / cols);
    }

    public static void varyingModelPatchTST(int error_rate) throws Exception {
        recordFile("\nERROR RATE " + error_rate + " (PatchTST)\n", "RMSE");
        recordFile("\nERROR RATE " + error_rate + " (PatchTST)\n", "Time");
        recordFile("\nERROR RATE " + error_rate + " (PatchTST)\n", "Precision");
        recordFile("\nERROR RATE " + error_rate + " (PatchTST)\n", "Recall");
        recordFile("\nERROR RATE " + error_rate + " (PatchTST)\n", "RegressionLoss");
        String data_csv_path = OUTPUT_BASE_PATH;
        String[] datasets = DATASETS;
        for (int dataset_idx = 0; dataset_idx < 5; dataset_idx++) {
            String file_path = data_csv_path + datasets[dataset_idx] + "/";
            File directory = new File(file_path);
            if (!directory.exists()) {
                directory.mkdirs();
            }

            init(dataset_idx);

            LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
            long[] td_time = loadData.getTd_time();
            String [] td_time_str = loadData.getTd_time_str();
            double[][] td_clean = loadData.getTd_clean();
            double[][] md = loadData.getMd();
            KDTreeUtil kdTree = loadData.getKdTree();
            KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
//            System.out.println(Arrays.toString(td_time));
            writeDataToCSV(td_time_str, td_clean, file_path + "clean_t.csv");

            // detect ground truth
            boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

            // add noise
            AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
            double[][] td_dirty = addNoise.getTd_dirty();
            writeDataToCSV(td_time_str, td_dirty, file_path + "dirty_" + error_rate + "_t.csv");

            // label4imr
            LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
            boolean[] td_bool = labelData.getTd_bool();

            recordFile("\n" + datasets[dataset_idx] + ",", "RMSE");
            recordFile("\n" + datasets[dataset_idx] + ",", "Precision");
            recordFile("\n" + datasets[dataset_idx] + ",", "Recall");
            recordFile("\n" + datasets[dataset_idx] + ",", "Time");
            recordFile("\n" + datasets[dataset_idx] + ",", "RegressionLoss");

            System.out.println("\nARMRepair (PatchTST)");
            System.out.println("Dataset: " + datasets[dataset_idx]);
            int columnCnt = td_clean[0].length;
            try {
                // create PatchTST model
                System.out.println("[Experiment] Creating PatchTST model for dataset: " + datasets[dataset_idx]);
                PatchTSTUtil patchTSTModel = new PatchTSTUtil(p, columnCnt);
                // create ARMRepair with PatchTST model
                System.out.println("[Experiment] Starting ARMRepair repair process...");
                ARMRepair armRepair = new ARMRepair(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta, patchTSTModel);
                System.out.println("[Experiment] Repair process completed successfully");
                
                double[][] td_repair_mr = armRepair.getTd_repaired();
                writeDataToCSV(td_time_str, td_repair_mr, file_path + "ARM_PatchTST_t.csv");
                System.out.println("[Experiment] Repaired data written to CSV");

                long cost_time = armRepair.getCost_time();
                // run anomaly detection with PatchTST model
                System.out.println("[Experiment] Starting anomaly detection on repaired data...");
                boolean[] detect_repair = AnomalyDetector.detect(td_repair_mr, p, beta, patchTSTModel);
                System.out.println("[Experiment] Anomaly detection completed");
                
                System.out.println("[Experiment] Creating Analysis object...");
                Analysis analysis = new Analysis(td_time, td_clean, td_repair_mr, td_bool, cost_time, detect_clean, detect_repair);
                double regression_loss = armRepair.getRegression_loss();
                
                // print results for debugging
                System.out.println("[Experiment] Results for " + datasets[dataset_idx] + ":");
                System.out.println("  RMSE: " + analysis.getRMSE());
                System.out.println("  Precision: " + analysis.getPrecision());
                System.out.println("  Recall: " + analysis.getRecall());
                System.out.println("  Time: " + analysis.getCost_time() + " ms");
                System.out.println("  Rounds: " + armRepair.getActual_rounds() + "/" + armRepair.getT());
                System.out.println("  Regression Loss: " + regression_loss);
                
                // write results to files
                System.out.println("[Experiment] Writing results to files...");
                recordFile(analysis.getRMSE() + ",", "RMSE");
                recordFile(analysis.getPrecision() + ",", "Precision");
                recordFile(analysis.getRecall() + ",", "Recall");
                recordFile(analysis.getCost_time() + ",", "Time");
                recordFile(regression_loss + ",", "RegressionLoss");
                System.out.println("[Experiment] Results written successfully");
            } catch (Exception e) {
                System.err.println("[Experiment] Error processing dataset " + datasets[dataset_idx] + ":");
                e.printStackTrace();
                // write placeholder on error to keep file format consistent
                recordFile("ERROR,", "RMSE");
                recordFile("ERROR,", "Precision");
                recordFile("ERROR,", "Recall");
                recordFile("ERROR,", "Time");
                recordFile("ERROR,", "RegressionLoss");
                throw e;  // rethrow for debugging
            }

            System.gc();
            Runtime.getRuntime().gc();
        }
    }

    public static void varyingModelDLinear(int error_rate) throws Exception {
        recordFile("\nERROR RATE " + error_rate + " (ARM+DLinear forecast)\n", "RMSE");
        recordFile("\nERROR RATE " + error_rate + " (ARM+DLinear forecast) Time=train,repair,pred\n", "Time");
        recordFile("\nERROR RATE " + error_rate + " (ARM+DLinear forecast)\n", "Precision");
        recordFile("\nERROR RATE " + error_rate + " (ARM+DLinear forecast)\n", "Recall");
        recordFile("\nERROR RATE " + error_rate + " (ARM+DLinear forecast)\n", "RegressionLoss");
        String data_csv_path = OUTPUT_BASE_PATH;
        String[] datasets = DATASETS;
        final double trainRatio = 0.8;

        for (int dataset_idx = 2; dataset_idx < 3; dataset_idx++) {
            String file_path = data_csv_path + datasets[dataset_idx] + "/";
            File directory = new File(file_path);
            if (!directory.exists()) {
                directory.mkdirs();
            }

            init(dataset_idx);

            LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
            long[] td_time = loadData.getTd_time();
            String[] td_time_str = loadData.getTd_time_str();
            double[][] td_clean = loadData.getTd_clean();
            KDTreeUtil kdTree = loadData.getKdTree();
            KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
            writeDataToCSV(td_time_str, td_clean, file_path + "clean_t.csv");

            AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
            double[][] td_dirty = addNoise.getTd_dirty();
            writeDataToCSV(td_time_str, td_dirty, file_path + "dirty_" + error_rate + "_t.csv");

            int n = td_dirty.length;
            int columnCnt = td_clean[0].length;
            int trainLen = (int) (n * trainRatio);
            if (trainLen <= p || trainLen >= n) {
                throw new IllegalStateException("Invalid 80/20 split for " + datasets[dataset_idx]);
            }

            recordFile("\n" + datasets[dataset_idx] + ",", "RMSE");
            // recordFile("\n" + datasets[dataset_idx] + ",", "Precision");
            // recordFile("\n" + datasets[dataset_idx] + ",", "Recall");
            recordFile("\n" + datasets[dataset_idx] + ",", "Time");
            // recordFile("\n" + datasets[dataset_idx] + ",", "RegressionLoss");

            System.out.println("\n===== ARM+DLinear forecast eval: " + datasets[dataset_idx] + " =====");
            System.out.println("n=" + n + ", trainLen(80%)=" + trainLen + ", testLen(20%)=" + (n - trainLen));

            try {
                System.out.println("[Experiment] ARM-repair with DLinear; train model on conflict-free points in first 80%...");
                DLinearUtil dlinearModel = new DLinearUtil(p, columnCnt);
                ARMRepair armRepair = new ARMRepair(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta, 1, dlinearModel, trainLen);
                double[][] td_repaired = armRepair.getTd_repaired();
                TimeSeriesPredictor model = armRepair.getPrediction_model();
                writeDataToCSV(td_time_str, td_repaired, file_path + "ARM_DLinear_repaired.csv");

                long predStart = System.nanoTime();
                double[][] predDirty = rollingPredictFrom(model, td_dirty, trainLen, p);
                long predMs = (System.nanoTime() - predStart) / 1_000_000L;
                double rmseDirty = rmseAgainstClean(predDirty, td_clean, trainLen);
                double[][] predRepaired = rollingPredictFrom(model, td_repaired, trainLen, p);
                double rmseRepaired = rmseAgainstClean(predRepaired, td_clean, trainLen);

                long trainMs = armRepair.getModel_train_time();
                long repairMs = armRepair.getRepair_time();

                writeForecastCompareCSV(td_time_str, td_clean, td_dirty, td_repaired,
                        predDirty, predRepaired, trainLen, file_path + "ARM_DLinear_forecast_compare.csv");

                System.out.println("Results for " + datasets[dataset_idx] + ":");
                System.out.println("  RMSE (model on dirty test):     " + String.format("%.3f", rmseDirty));
                System.out.println("  RMSE (model on repaired test):  " + String.format("%.3f", rmseRepaired));
                System.out.println("  Time (ms): train=" + trainMs
                        + ", repair=" + repairMs
                        + ", pred=" + predMs
                        + "; ARM rounds: " + armRepair.getActual_rounds() + "/" + armRepair.getT());

                recordFile(String.format("%.3f", rmseDirty) + ",", "RMSE");
                recordFile(String.format("%.3f", rmseRepaired) + ",", "RMSE");
                recordFile(trainMs + ",", "Time");
                recordFile(repairMs + ",", "Time");
                recordFile(predMs + ",", "Time");
                // recordFile("NA,", "Precision");
                // recordFile("NA,", "Precision");
                // recordFile("NA,", "Recall");
                // recordFile("NA,", "Recall");
                // recordFile("NA,", "RegressionLoss");
                // recordFile("NA,", "RegressionLoss");
            } catch (Exception e) {
                System.err.println("[Experiment] Error processing dataset " + datasets[dataset_idx] + ":");
                e.printStackTrace();
                recordFile("ERROR,", "RMSE");
                recordFile("ERROR,", "RMSE");
                recordFile("ERROR,", "Time");
                recordFile("ERROR,", "Time");
                recordFile("ERROR,", "Time");
                // recordFile("ERROR,", "Precision");
                // recordFile("ERROR,", "Precision");
                // recordFile("ERROR,", "Recall");
                // recordFile("ERROR,", "Recall");
                // recordFile("ERROR,", "RegressionLoss");
                // recordFile("ERROR,", "RegressionLoss");
                throw e;
            }

            System.gc();
            Runtime.getRuntime().gc();
        }
    }

    /** Deep-copy rows [start, end). */
    private static double[][] sliceRows(double[][] data, int start, int end) {
        double[][] out = new double[end - start][];
        for (int i = start; i < end; i++) {
            out[i - start] = data[i].clone();
        }
        return out;
    }

    private static ArrayList<ArrayList<Double>> toSamples(double[][] data) {
        ArrayList<ArrayList<Double>> samples = new ArrayList<>(data.length);
        for (double[] row : data) {
            ArrayList<Double> sample = new ArrayList<>(row.length);
            for (double v : row) {
                sample.add(v);
            }
            samples.add(sample);
        }
        return samples;
    }

    /**
     * Count points that pass domain-constraint consistency check — same training size as ARM repair.
     */
    private static int countNormalPoints(double[][] data, KDTreeUtil kdTree, double eta) {
        double[] std = computeStd(data);
        int count = 0;
        for (double[] tuple : data) {
            double[] nn = kdTree.nearestNeighbor(tuple);
            if (!(signedDelta(tuple, nn, std) > eta)) {
                count++;
            }
        }
        return count;
    }

    private static double[] computeStd(double[][] data) {
        int n = data.length;
        int cols = data[0].length;
        double[] std = new double[cols];
        for (int c = 0; c < cols; c++) {
            double sum = 0.0;
            int cnt = 0;
            for (int i = 0; i < n; i++) {
                if (!Double.isNaN(data[i][c])) {
                    sum += data[i][c];
                    cnt++;
                }
            }
            double mean = cnt == 0 ? 0.0 : sum / cnt;
            double var = 0.0;
            for (int i = 0; i < n; i++) {
                if (!Double.isNaN(data[i][c])) {
                    double d = data[i][c] - mean;
                    var += d * d;
                }
            }
            std[c] = cnt == 0 ? 1.0 : Math.sqrt(var / cnt);
            if (std[c] < 1e-8) {
                std[c] = 1.0;
            }
        }
        return std;
    }

    /** Same signed normalized delta as ARMRepair.delta */
    private static double signedDelta(double[] a, double[] b, double[] std) {
        double distance = 0.0;
        for (int i = 0; i < a.length; i++) {
            distance += (a[i] - b[i]) / std[i];
        }
        return distance;
    }

    /**
     * Rolling one-step prediction on indices [windowSize, n).
     */
    private static double[][] rollingPredictFull(TimeSeriesPredictor model, double[][] context, int windowSize) {
        return rollingPredictFrom(model, context, windowSize, windowSize);
    }

    /**
     * Rolling one-step prediction on indices [startIdx, n).
     * {@code startIdx} must be &gt;= {@code windowSize}.
     */
    private static double[][] rollingPredictFrom(TimeSeriesPredictor model, double[][] context,
                                                 int startIdx, int windowSize) {
        if (startIdx < windowSize) {
            throw new IllegalArgumentException("startIdx must be >= windowSize");
        }
        int n = context.length;
        int cols = context[0].length;
        double[][] preds = new double[n - startIdx][cols];
        for (int i = startIdx; i < n; i++) {
            double[][] window = new double[windowSize][cols];
            for (int j = 0; j < windowSize; j++) {
                window[j] = context[i - windowSize + j].clone();
            }
            ArrayList<Double> pred = model.predict(window);
            for (int c = 0; c < cols; c++) {
                preds[i - startIdx][c] = pred.get(c);
            }
        }
        return preds;
    }

    /** RMSE of predictions against clean ground truth on [startIdx, n). */
    private static double rmseAgainstClean(double[][] preds, double[][] clean, int startIdx) {
        int len = preds.length;
        int cols = preds[0].length;
        double sse = 0.0;
        for (int i = 0; i < len; i++) {
            for (int c = 0; c < cols; c++) {
                double diff = clean[startIdx + i][c] - preds[i][c];
                sse += diff * diff;
            }
        }
        return Math.sqrt(sse / len / cols);
    }

    public static double[][] loadCsvFile(String filePath) throws FileNotFoundException {
        ArrayList<ArrayList<Double>> data = new ArrayList<>();
        Scanner sc = new Scanner(new File(filePath));
        sc.useDelimiter("\\s*([,\\r\\n])\\s*"); // set separator
        sc.nextLine();  // skip table header
        for (int k = td_len; k > 0 && sc.hasNextLine(); --k) {  // the size of td_clean is dataLen
            String new_line = sc.nextLine();

            if (new_line.charAt(new_line.length()-1) == ',') {
                new_line = new_line + '0';
            }
            String[] line_str = (new_line).split(",");

            addValues(data, line_str);
        }


        double[][] data_array = getDoubleArray(data);
        return data_array;
    }

    private static void addValues(ArrayList<ArrayList<Double>> array, String[] line) {
        ArrayList<Double> values = new ArrayList<>();
        String value;
        for (int i = 1; i < line.length; ++i) {
            value = line[i];
            if (!value.equals("")) {
                values.add(Double.parseDouble(value));
            } else {
                values.add(Double.NaN);
            }
        }
        array.add(values);
    }

    private static double[][] getDoubleArray(ArrayList<ArrayList<Double>> arrayList) {
        double[][] rtn = new double[arrayList.size()][arrayList.get(0).size()];
        for (int i = 0, j; i < arrayList.size(); ++i)
            for (j = 0; j < arrayList.get(0).size(); ++j)
                rtn[i][j] = arrayList.get(i).get(j);
        return rtn;
    }

    public static double[][] trimToShortestLength(double[][] list1, double[][] list2) {
        int minLength = Math.min(list1.length, list2.length);
        double[][] trimmedList = new double[minLength][];
        System.arraycopy(list1, 0, trimmedList, 0, minLength);
        return trimmedList;
    }


    // -------------------------------------------------------------------------
    // 7. End-to-end repair benchmarks
    // -------------------------------------------------------------------------

    /**
     * ARM-only evaluation across datasets for a given error rate.
     * Runs 2 trials with different seeds; writes mean ± sample std.
     */
    public static void get_arm_repaired(int rate) throws Exception {
        error_rate = rate;
        String data_csv_path = DATA_BASE_PATH + "error_rate_" + String.valueOf(rate) + "/";
        String[] datasets = DATASETS;
        int nDatasets = datasets.length;
        final int nTrials = 2;
        final int baseSeed = 665;

        File resultsDir = new File("./results/arm_performance_v2");
        if (!resultsDir.exists()) {
            resultsDir.mkdirs();
        }
        String rmseCsv = "./results/arm_performance_v2/rmse.csv";
        String timeCsv = "./results/arm_performance_v2/time.csv";

        try (BufferedWriter rmseOut = new BufferedWriter(new FileWriter(rmseCsv));
             BufferedWriter timeOut = new BufferedWriter(new FileWriter(timeCsv))) {
            rmseOut.write("dataset,ARM");
            rmseOut.newLine();
            timeOut.write("dataset,ARM");
            timeOut.newLine();

            System.out.println("\n===== ARM-only, " + nTrials + " trials (mean ± std) =====");
            for (int dataset_idx = 0; dataset_idx < nDatasets; dataset_idx++) {
                String file_path = data_csv_path + datasets[dataset_idx] + "/";
                File directory = new File(file_path);
                if (!directory.exists()) {
                    directory.mkdirs();
                }
                init(dataset_idx);

                LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, baseSeed);
                long[] td_time = loadData.getTd_time();
                double[][] td_clean = loadData.getTd_clean();
                KDTreeUtil kdTree = loadData.getKdTree();
                KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
                boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);
                int columnCnt = td_clean[0].length;

                double[] rmses = new double[nTrials];
                double[] times = new double[nTrials];
                System.out.println("\n===== ARM-only dataset: " + datasets[dataset_idx] + " =====");
                for (int t = 0; t < nTrials; t++) {
                    seed = baseSeed + t;
                    double[][] td_dirty = new AddNoise(td_clean, error_rate, error_range, error_length,
                            eta, kdTreeComplete, seed).getTd_dirty();
                    LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
                    ARMRepair armRepair = new ARMRepair(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta,
                            3, new VARUtil(p));
                    double[][] td_repair_mr = armRepair.getTd_repaired();
                    boolean[] detect_repair = AnomalyDetector.detect(td_repair_mr, p, beta);
                    Analysis analysis = new Analysis(td_time, td_clean, td_repair_mr, labelData.getTd_bool(),
                            armRepair.getCost_time(), detect_clean, detect_repair);
                    rmses[t] = analysis.getRMSEValue();
                    times[t] = analysis.getCost_time();
                    System.out.println("  trial " + (t + 1) + "/" + nTrials
                            + " seed=" + seed
                            + " RMSE=" + String.format("%.6f", rmses[t])
                            + " Time=" + (long) times[t] + " ms"
                            + " Rounds=" + armRepair.getActual_rounds() + "/" + armRepair.getT());
                    System.gc();
                    Runtime.getRuntime().gc();
                }
                String rmseCell = formatMeanStd(rmses);
                String timeCell = formatMeanStd(times);
                rmseOut.write(datasets[dataset_idx] + "," + rmseCell);
                rmseOut.newLine();
                timeOut.write(datasets[dataset_idx] + "," + timeCell);
                timeOut.newLine();
                System.out.println("  RMSE: " + rmseCell);
                System.out.println("  Time (ms): " + timeCell);
            }
        }
        System.out.println("Wrote RMSE CSV: " + rmseCsv);
        System.out.println("Wrote Time CSV: " + timeCsv);
    }

    /**
     * DomainNN + ModelOnly only, same trial protocol as {@link #get_arm_repaired}.
     */
    public static void get_baseline_repaired(int rate) throws Exception {
        error_rate = rate;
        String[] datasets = DATASETS;
        String[] methods = {"DomainNN", "ModelOnly"};
        int nDatasets = datasets.length;
        final int nTrials = 2;
        final int baseSeed = 665;

        File resultsDir = new File("./results/baseline_performance");
        if (!resultsDir.exists()) {
            resultsDir.mkdirs();
        }
        String rmseCsv = "./results/baseline_performance/rmse.csv";
        String timeCsv = "./results/baseline_performance/time.csv";

        try (BufferedWriter rmseOut = new BufferedWriter(new FileWriter(rmseCsv));
             BufferedWriter timeOut = new BufferedWriter(new FileWriter(timeCsv))) {
            rmseOut.write("dataset,DomainNN,ModelOnly");
            rmseOut.newLine();
            timeOut.write("dataset,DomainNN,ModelOnly");
            timeOut.newLine();

            System.out.println("\n===== DomainNN + ModelOnly, " + nTrials + " trials (mean ± std) =====");
            for (int dataset_idx = 0; dataset_idx < nDatasets; dataset_idx++) {
                init(dataset_idx);
                LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, baseSeed);
                long[] td_time = loadData.getTd_time();
                double[][] td_clean = loadData.getTd_clean();
                KDTreeUtil kdTree = loadData.getKdTree();
                KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
                boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

                System.out.println("\n===== baselines dataset: " + datasets[dataset_idx] + " =====");
                double[][] rmses = new double[methods.length][nTrials];
                double[][] times = new double[methods.length][nTrials];

                for (int t = 0; t < nTrials; t++) {
                    seed = baseSeed + t;
                    double[][] td_dirty = new AddNoise(td_clean, error_rate, error_range, error_length,
                            eta, kdTreeComplete, seed).getTd_dirty();
                    boolean[] td_bool = new LabelData(td_clean, td_dirty, label_rate, seed).getTd_bool();

                    Analysis domain = domainNnRepair(kdTree, td_time, td_clean, td_dirty, td_bool, detect_clean);
                    rmses[0][t] = domain.getRMSEValue();
                    times[0][t] = domain.getCost_time();
                    System.out.println("  trial " + (t + 1) + "/" + nTrials + " DomainNN"
                            + " seed=" + seed
                            + " RMSE=" + String.format("%.6f", rmses[0][t])
                            + " Time=" + (long) times[0][t] + " ms");
                    gcQuiet();

                    Analysis model = modelOnlyRepair(td_time, td_clean, td_dirty, td_bool, detect_clean);
                    rmses[1][t] = model.getRMSEValue();
                    times[1][t] = model.getCost_time();
                    System.out.println("  trial " + (t + 1) + "/" + nTrials + " ModelOnly"
                            + " seed=" + seed
                            + " RMSE=" + String.format("%.6f", rmses[1][t])
                            + " Time=" + (long) times[1][t] + " ms");
                    gcQuiet();
                }

                StringBuilder rmseLine = new StringBuilder(datasets[dataset_idx]);
                StringBuilder timeLine = new StringBuilder(datasets[dataset_idx]);
                for (int m = 0; m < methods.length; m++) {
                    String rmseCell = formatMeanStd(rmses[m]);
                    String timeCell = formatMeanStd(times[m]);
                    rmseLine.append(',').append(rmseCell);
                    timeLine.append(',').append(timeCell);
                    System.out.println("  " + methods[m] + " RMSE: " + rmseCell);
                    System.out.println("  " + methods[m] + " Time (ms): " + timeCell);
                }
                rmseOut.write(rmseLine.toString());
                rmseOut.newLine();
                rmseOut.flush();
                timeOut.write(timeLine.toString());
                timeOut.newLine();
                timeOut.flush();
            }
        }
        System.out.println("Wrote RMSE CSV: " + rmseCsv);
        System.out.println("Wrote Time CSV: " + timeCsv);
    }

    /**
     * ARM vs CSDI on the same prefix of each dataset. CSDI self-detects then
     * imputes; it does not receive ARM's error set E.
     */
    public static void get_csdi_repaired(int rate) throws Exception {
        error_rate = rate;
        String[] datasets = DATASETS;
        String[] methods = {"ARM", "CSDI"};
        int nDatasets = datasets.length;
        final int nTrials = 1;
        final int baseSeed = 665;

        File resultsDir = new File("./results/csdi_performance");
        if (!resultsDir.exists()) {
            resultsDir.mkdirs();
        }
        String rmseCsv = "./results/csdi_performance/rmse.csv";
        String timeCsv = "./results/csdi_performance/time.csv";

        try (BufferedWriter rmseOut = new BufferedWriter(new FileWriter(rmseCsv));
             BufferedWriter timeOut = new BufferedWriter(new FileWriter(timeCsv))) {
            rmseOut.write("dataset,ARM,CSDI");
            rmseOut.newLine();
            timeOut.write("dataset,ARM,CSDI");
            timeOut.newLine();

            System.out.println("\n===== ARM vs CSDI (self-detect then impute), prefix="
                    + CSDI_COMPARE_LEN + " =====");
            for (int dataset_idx = 0; dataset_idx < nDatasets; dataset_idx++) {
                init(dataset_idx);
                LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, baseSeed);
                long[] td_time_full = loadData.getTd_time();
                double[][] td_clean_full = loadData.getTd_clean();
                KDTreeUtil kdTree = loadData.getKdTree();
                KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();

                int n = Math.min(CSDI_COMPARE_LEN, td_clean_full.length);
                long[] td_time = Arrays.copyOf(td_time_full, n);
                double[][] td_clean = sliceRows(td_clean_full, 0, n);
                boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

                System.out.println("\n===== CSDI dataset: " + datasets[dataset_idx]
                        + " n=" + n + " =====");
                double[][] rmses = new double[methods.length][nTrials];
                double[][] times = new double[methods.length][nTrials];

                for (int t = 0; t < nTrials; t++) {
                    seed = baseSeed + t;
                    double[][] td_dirty_full = new AddNoise(td_clean_full, error_rate, error_range,
                            error_length, eta, kdTreeComplete, seed).getTd_dirty();
                    boolean[] td_bool = new LabelData(td_clean, sliceRows(td_dirty_full, 0, n),
                            label_rate, seed).getTd_bool();

                    Analysis arm = armRepair(kdTree, td_time, td_clean,
                            sliceRows(td_dirty_full, 0, n), td_bool, detect_clean);
                    rmses[0][t] = arm.getRMSEValue();
                    times[0][t] = arm.getCost_time();
                    System.out.println("  trial " + (t + 1) + "/" + nTrials + " ARM"
                            + " RMSE=" + String.format("%.6f", rmses[0][t])
                            + " Time=" + (long) times[0][t] + " ms");
                    gcQuiet();

                    Analysis csdi = csdiRepair(td_time, td_clean,
                            sliceRows(td_dirty_full, 0, n), td_bool, detect_clean);
                    rmses[1][t] = csdi.getRMSEValue();
                    times[1][t] = csdi.getCost_time();
                    System.out.println("  trial " + (t + 1) + "/" + nTrials + " CSDI"
                            + " RMSE=" + String.format("%.6f", rmses[1][t])
                            + " Time=" + (long) times[1][t] + " ms");
                    gcQuiet();
                }

                StringBuilder rmseLine = new StringBuilder(datasets[dataset_idx]);
                StringBuilder timeLine = new StringBuilder(datasets[dataset_idx]);
                for (int m = 0; m < methods.length; m++) {
                    String rmseCell = formatMeanStd(rmses[m]);
                    String timeCell = formatMeanStd(times[m]);
                    rmseLine.append(',').append(rmseCell);
                    timeLine.append(',').append(timeCell);
                    System.out.println("  " + methods[m] + " RMSE: " + rmseCell);
                    System.out.println("  " + methods[m] + " Time (ms): " + timeCell);
                }
                rmseOut.write(rmseLine.toString());
                rmseOut.newLine();
                rmseOut.flush();
                timeOut.write(timeLine.toString());
                timeOut.newLine();
                timeOut.flush();
            }
        }
        System.out.println("Wrote RMSE CSV: " + rmseCsv);
        System.out.println("Wrote Time CSV: " + timeCsv);
    }

    /**
     * Precision / recall of ARM's two initial detectors vs injected errors
     * (row dirty ≠ clean). No repair, no other baselines.
     */
    public static void get_detection_pr() throws Exception {
        String[] datasets = DATASETS;
        File resultsDir = new File("./results/detection_pr");
        if (!resultsDir.exists()) {
            resultsDir.mkdirs();
        }
        String csv = "./results/detection_pr/pr.csv";

        try (BufferedWriter out = new BufferedWriter(new FileWriter(csv))) {
            out.write("dataset,n,n_true,e_domain_size,e_domain_precision,e_domain_recall,"
                    + "e_model_size,e_model_precision,e_model_recall");
            out.newLine();

            System.out.println("\n===== E_domain / E_model precision & recall =====");
            for (int dataset_idx = 0; dataset_idx < datasets.length; dataset_idx++) {
                init(dataset_idx);
                LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
                double[][] td_clean = loadData.getTd_clean();
                KDTreeUtil kdTree = loadData.getKdTree();
                KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
                long[] td_time = loadData.getTd_time();
                int columnCnt = td_clean[0].length;

                double[][] td_dirty = new AddNoise(td_clean, error_rate, error_range, error_length,
                        eta, kdTreeComplete, seed).getTd_dirty();
                boolean[] gold = trueErrorMask(td_clean, td_dirty);

                ARMRepair detector = ARMRepair.forDetection(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta);
                boolean[] eDomain = detector.getE_domain();
                boolean[] eModel = detector.getE_model();

                double[] domainPr = precisionRecall(eDomain, gold);
                double[] modelPr = precisionRecall(eModel, gold);
                int nTrue = countTrue(gold);
                int nDom = countTrue(eDomain);
                int nMod = countTrue(eModel);

                System.out.println("\n----- " + datasets[dataset_idx]
                        + " n=" + td_clean.length
                        + " n_true=" + nTrue + " -----");
                System.out.println("  E_domain |E|=" + nDom
                        + " P=" + String.format("%.6f", domainPr[0])
                        + " R=" + String.format("%.6f", domainPr[1])
                        + " tp/fp/fn=" + (int) domainPr[2] + "/" + (int) domainPr[3] + "/" + (int) domainPr[4]);
                System.out.println("  E_model  |E|=" + nMod
                        + " P=" + String.format("%.6f", modelPr[0])
                        + " R=" + String.format("%.6f", modelPr[1])
                        + " tp/fp/fn=" + (int) modelPr[2] + "/" + (int) modelPr[3] + "/" + (int) modelPr[4]);

                out.write(datasets[dataset_idx]
                        + "," + td_clean.length
                        + "," + nTrue
                        + "," + nDom
                        + "," + String.format("%.6f", domainPr[0])
                        + "," + String.format("%.6f", domainPr[1])
                        + "," + nMod
                        + "," + String.format("%.6f", modelPr[0])
                        + "," + String.format("%.6f", modelPr[1]));
                out.newLine();
                out.flush();
                gcQuiet();
            }
        }
        System.out.println("Wrote P/R CSV: " + csv);
    }

    /** Ground truth: a timestep is an error if any variable differs from clean. */
    static boolean[] trueErrorMask(double[][] clean, double[][] dirty) {
        boolean[] gold = new boolean[clean.length];
        for (int i = 0; i < clean.length; i++) {
            for (int c = 0; c < clean[i].length; c++) {
                if (clean[i][c] != dirty[i][c]) {
                    gold[i] = true;
                    break;
                }
            }
        }
        return gold;
    }

    /**
     * @return {precision, recall, tp, fp, fn}; precision/recall are 0 when the
     *         denominator is 0 (no Laplace smoothing).
     */
    static double[] precisionRecall(boolean[] pred, boolean[] gold) {
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < gold.length; i++) {
            if (pred[i] && gold[i]) {
                tp++;
            } else if (pred[i]) {
                fp++;
            } else if (gold[i]) {
                fn++;
            }
        }
        double precision = (tp + fp) == 0 ? 0.0 : tp / (double) (tp + fp);
        double recall = (tp + fn) == 0 ? 0.0 : tp / (double) (tp + fn);
        return new double[]{precision, recall, tp, fp, fn};
    }

    static int countTrue(boolean[] flags) {
        int n = 0;
        for (boolean f : flags) {
            if (f) {
                n++;
            }
        }
        return n;
    }

    /**
     * Inject Gaussian noise into a fraction of domain-constraint rows, rebuild the
     * ARM k-NN index, then repair. Road dataset + ARM only.
     * Percents: 20, 40, 60, 80, 100. Time-series dirt is injected against the
     * original (clean) domain so only constraint quality varies.
     */
    public static void get_domain_noise_arm(int rate) throws Exception {
        error_rate = rate;
        final int dataset_idx = 2; // road
        final int nTrials = 2;
        final int baseSeed = 665;
        int[] percents = {20, 40, 60, 80, 100};

        init(dataset_idx);
        LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, baseSeed);
        long[] td_time = loadData.getTd_time();
        double[][] td_clean = loadData.getTd_clean();
        double[][] mdClean = sliceRows(loadData.getMd(), 0, loadData.getMd().length);
        KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
        boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);
        int columnCnt = td_clean[0].length;

        File resultsDir = new File("./results/domain_noise/road");
        if (!resultsDir.exists()) {
            resultsDir.mkdirs();
        }
        String rmseCsv = "./results/domain_noise/road/rmse.csv";
        String timeCsv = "./results/domain_noise/road/time.csv";

        System.out.println("\n===== Domain-noise ARM, road, " + nTrials + " trials =====");
        System.out.println("series error_rate=" + error_rate + ", error_range=" + error_range
                + ", md_len=" + mdClean.length);

        try (BufferedWriter rmseOut = new BufferedWriter(new FileWriter(rmseCsv));
             BufferedWriter timeOut = new BufferedWriter(new FileWriter(timeCsv))) {
            rmseOut.write("domain_noise_pct,rmse");
            rmseOut.newLine();
            timeOut.write("domain_noise_pct,time_ms");
            timeOut.newLine();

            for (int pct : percents) {
                double[] rmses = new double[nTrials];
                double[] times = new double[nTrials];
                System.out.println("\n----- domain noise " + pct + "% -----");
                for (int t = 0; t < nTrials; t++) {
                    seed = baseSeed + t;
                    double[][] td_dirty = new AddNoise(td_clean, error_rate, error_range, error_length,
                            eta, kdTreeComplete, seed).getTd_dirty();
                    boolean[] td_bool = new LabelData(td_clean, td_dirty, label_rate, seed).getTd_bool();

                    double[][] mdNoisy = corruptDomainConstraints(mdClean, pct, error_range, seed);
                    KDTreeUtil kdNoisy = new KDTreeUtil(sliceRows(mdNoisy, 0, mdNoisy.length));

                    ARMRepair armRepair = new ARMRepair(td_dirty, kdNoisy, td_time, columnCnt, k, p, eta, beta,
                            3, new VARUtil(p));
                    Analysis analysis = new Analysis(td_time, td_clean, armRepair.getTd_repaired(),
                            td_bool, armRepair.getCost_time(), detect_clean,
                            AnomalyDetector.detect(armRepair.getTd_repaired(), p, beta));
                    rmses[t] = analysis.getRMSEValue();
                    times[t] = analysis.getCost_time();
                    System.out.println("  trial " + (t + 1) + "/" + nTrials + " seed=" + seed
                            + " RMSE=" + String.format("%.6f", rmses[t])
                            + " Time=" + (long) times[t] + " ms"
                            + " Rounds=" + armRepair.getActual_rounds() + "/" + armRepair.getT());
                    gcQuiet();
                }
                String rmseCell = formatMeanStd(rmses);
                String timeCell = formatMeanStd(times);
                rmseOut.write(pct + "," + rmseCell);
                rmseOut.newLine();
                rmseOut.flush();
                timeOut.write(pct + "," + timeCell);
                timeOut.newLine();
                timeOut.flush();
                System.out.println("  ARM RMSE: " + rmseCell);
                System.out.println("  ARM Time (ms): " + timeCell);
            }
        }
        System.out.println("Wrote RMSE CSV: " + rmseCsv);
        System.out.println("Wrote Time CSV: " + timeCsv);
    }

    /**
     * Corrupt {@code percent}% of domain-constraint rows with N(0,1)*range (same
     * magnitude as series error_range). Returns a new array; input is not modified.
     */
    private static double[][] corruptDomainConstraints(double[][] md, int percent, double range, int seed) {
        double[][] noisy = sliceRows(md, 0, md.length);
        if (percent <= 0) {
            return noisy;
        }
        int n = noisy.length;
        int nCorrupt = (int) Math.round(n * Math.min(percent, 100) / 100.0);
        Random rng = new Random(seed + 17 * percent);
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        for (int i = 0; i < nCorrupt; i++) {
            int j = i + rng.nextInt(n - i);
            int tmp = idx[i];
            idx[i] = idx[j];
            idx[j] = tmp;
        }
        int cols = noisy[0].length;
        for (int i = 0; i < nCorrupt; i++) {
            double g = rng.nextGaussian();
            int row = idx[i];
            for (int c = 0; c < cols; c++) {
                noisy[row][c] += g * range;
            }
        }
        System.out.println("  corrupted " + nCorrupt + " / " + n + " domain rows (" + percent + "%)");
        return noisy;
    }

    private static String formatMeanStd(double[] values) {
        double mean = 0.0;
        for (double v : values) {
            mean += v;
        }
        mean /= values.length;
        double var = 0.0;
        for (double v : values) {
            double d = v - mean;
            var += d * d;
        }
        double std = values.length > 1 ? Math.sqrt(var / (values.length - 1)) : 0.0;
        return String.format("%.6f ± %.6f", mean, std);
    }

    /**
     * Inject a contiguous error burst of given length at the start of the road
     * series, then overlay {@link AddNoise} (init error_rate, per-mille). ARM only.
     * Lengths: 5, 10, 15, 20, 25.
     */
    public static void get_prefix_error_arm() throws Exception {
        final int dataset_idx = 2; // road
        int[] lengths = {5, 10, 15, 20, 25};
        init(dataset_idx);

        LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
        long[] td_time = loadData.getTd_time();
        double[][] td_clean = loadData.getTd_clean();
        KDTreeUtil kdTree = loadData.getKdTree();
        KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
        boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);
        int columnCnt = td_clean[0].length;
        boolean[] td_bool = new boolean[td_clean.length];

        File resultsDir = new File("./results/prefix_error/road");
        if (!resultsDir.exists()) {
            resultsDir.mkdirs();
        }
        String rmseCsv = "./results/prefix_error/road/rmse.csv";
        String timeCsv = "./results/prefix_error/road/time.csv";

        System.out.println("\n===== Prefix-burst ARM, road =====");
        System.out.println("error_range=" + error_range + ", eta=" + eta + ", p=" + p
                + ", addNoise per-mille=" + error_rate
                + ", lengths=" + Arrays.toString(lengths));

        try (BufferedWriter rmseOut = new BufferedWriter(new FileWriter(rmseCsv));
             BufferedWriter timeOut = new BufferedWriter(new FileWriter(timeCsv))) {
            rmseOut.write("error_length,rmse");
            rmseOut.newLine();
            timeOut.write("error_length,time_ms");
            timeOut.newLine();

            for (int len : lengths) {
                double[][] td_prefix = AddNoise.prefixBurst(td_clean, len, error_range, eta, kdTreeComplete, seed);
                double[][] td_dirty = new AddNoise(td_prefix, error_rate, error_range, error_length,
                        eta, kdTreeComplete, seed).getTd_dirty();
                ARMRepair armRepair = new ARMRepair(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta,
                        3, new VARUtil(p));
                Analysis analysis = new Analysis(td_time, td_clean, armRepair.getTd_repaired(),
                        td_bool, armRepair.getCost_time(), detect_clean,
                        AnomalyDetector.detect(armRepair.getTd_repaired(), p, beta));
                String rmse = String.format("%.6f", analysis.getRMSEValue());
                long timeMs = analysis.getCost_time();
                rmseOut.write(len + "," + rmse);
                rmseOut.newLine();
                rmseOut.flush();
                timeOut.write(len + "," + timeMs);
                timeOut.newLine();
                timeOut.flush();
                System.out.println("  prefix length=" + len
                        + " RMSE=" + rmse
                        + " Time=" + timeMs + " ms"
                        + " Rounds=" + armRepair.getActual_rounds() + "/" + armRepair.getT());
                gcQuiet();
            }
        }
        System.out.println("Wrote RMSE CSV: " + rmseCsv);
        System.out.println("Wrote Time CSV: " + timeCsv);
    }

    public static void get_data_repaired(int rate) throws Exception {
        error_rate = rate;
        String data_csv_path = DATA_BASE_PATH + "error_rate_" + String.valueOf(rate) + "/";
        String[] datasets = DATASETS;
        String[] methods = {"ARM", "DomainNN", "ModelOnly", "ER", "SCREEN", "Lsgreedy", "IMR", "MTCSC"};
        int nDatasets = datasets.length;
        int nMethods = methods.length;
        double[][] rmseTable = new double[nDatasets][nMethods];
        long[][] timeTable = new long[nDatasets][nMethods];

        for (int dataset_idx = 0; dataset_idx < nDatasets; dataset_idx++) {
            String file_path = data_csv_path + datasets[dataset_idx] + "/";
            File directory = new File(file_path);
            if (!directory.exists()) {
                directory.mkdirs();
            }
            init(dataset_idx);

            LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
            long[] td_time = loadData.getTd_time();
            double[][] td_clean = loadData.getTd_clean();
            double[][] md = loadData.getMd();
            KDTreeUtil kdTree = loadData.getKdTree();
            KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
//            writeDataToCSV(td_clean, file_path + "clean.csv");

            boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

            AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
            double[][] td_dirty = addNoise.getTd_dirty();
//            writeDataToCSV(td_dirty, file_path + "dirty.csv");

            LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
            double[][] td_label = labelData.getTd_label();
            boolean[] td_bool = labelData.getTd_bool();

//            recordFile("\n\n", "RMSE");
//            recordFile("\n\n", "Precision");
//            recordFile("\n\n", "Recall");
//            recordFile("\n\n", "Time");

            System.out.println("\n===== dataset: " + datasets[dataset_idx] + " =====");

            System.out.println("\nARMRepair");
            int columnCnt = td_clean[0].length;
            ARMRepair armRepair = new ARMRepair(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta);
            double[][] td_repair_mr = armRepair.getTd_repaired();
//            writeDataToCSV(td_repair_mr, file_path + "MR.csv");
            long cost_time = armRepair.getCost_time();
            boolean[] detect_repair = AnomalyDetector.detect(td_repair_mr, p, beta);
            Analysis analysis = new Analysis(td_time, td_clean, td_repair_mr, td_bool, cost_time, detect_clean, detect_repair);
            rmseTable[dataset_idx][0] = analysis.getRMSEValue();
            timeTable[dataset_idx][0] = analysis.getCost_time();
            System.gc();
            Runtime.getRuntime().gc();

            analysis = domainNnRepair(kdTree, td_time, td_clean, td_dirty, td_bool, detect_clean);
            rmseTable[dataset_idx][1] = analysis.getRMSEValue();
            timeTable[dataset_idx][1] = analysis.getCost_time();
            System.gc();
            Runtime.getRuntime().gc();

            analysis = modelOnlyRepair(td_time, td_clean, td_dirty, td_bool, detect_clean);
            rmseTable[dataset_idx][2] = analysis.getRMSEValue();
            timeTable[dataset_idx][2] = analysis.getCost_time();
            System.gc();
            Runtime.getRuntime().gc();

            System.out.println("\nERRepair");
            EditingRuleRepair editingRuleRepair = new EditingRuleRepair(td_time, td_dirty, md);
            double[][] td_repair_er = editingRuleRepair.getTd_repair();
//            writeDataToCSV(td_repair_er, file_path + "ER.csv");
            cost_time = editingRuleRepair.getCost_time();
            detect_repair = AnomalyDetector.detect(td_repair_er, p, beta);
            analysis = new Analysis(td_time, td_clean, td_repair_er, td_bool, cost_time, detect_clean, detect_repair);
            rmseTable[dataset_idx][3] = analysis.getRMSEValue();
            timeTable[dataset_idx][3] = analysis.getCost_time();
//            recordFile(analysis.getRMSE() + ",", "RMSE");
//            recordFile(analysis.getPrecision() + ",", "Precision");
//            recordFile(analysis.getRecall() + ",", "Recall");
//            recordFile(analysis.getCost_time() + ",", "Time");
            System.gc();
            Runtime.getRuntime().gc();

            System.out.println("\nSCREEN");
            SCREEN screen = new SCREEN(td_time, td_dirty);
            double[][] td_repair_sr = screen.getTd_repair();
//            writeDataToCSV(td_repair_sr, file_path + "SCREEN.csv");
            cost_time = screen.getCost_time();
            detect_repair = AnomalyDetector.detect(td_repair_sr, p, beta);
            analysis = new Analysis(td_time, td_clean, td_repair_sr, td_bool, cost_time, detect_clean, detect_repair);
            rmseTable[dataset_idx][4] = analysis.getRMSEValue();
            timeTable[dataset_idx][4] = analysis.getCost_time();
//            recordFile(analysis.getRMSE() + ",", "RMSE");
//            recordFile(analysis.getPrecision() + ",", "Precision");
//            recordFile(analysis.getRecall() + ",", "Recall");
//            recordFile(analysis.getCost_time() + ",", "Time");
            System.gc();
            Runtime.getRuntime().gc();

            System.out.println("\nLsgreedy");
            Lsgreedy lsgreedy = new Lsgreedy(td_time, td_dirty);
            double[][] td_repair_lg = lsgreedy.getTd_repair();
//            writeDataToCSV(td_repair_lg, file_path + "LG.csv");
            cost_time = lsgreedy.getCost_time();
            detect_repair = AnomalyDetector.detect(td_repair_lg, p, beta);
            analysis = new Analysis(td_time, td_clean, td_repair_lg, td_bool, cost_time, detect_clean, detect_repair);
            rmseTable[dataset_idx][5] = analysis.getRMSEValue();
            timeTable[dataset_idx][5] = analysis.getCost_time();
//            recordFile(analysis.getRMSE() + ",", "RMSE");
//            recordFile(analysis.getPrecision() + ",", "Precision");
//            recordFile(analysis.getRecall() + ",", "Recall");
//            recordFile(analysis.getCost_time() + ",", "Time");
            System.gc();
            Runtime.getRuntime().gc();

            System.out.println("\nIMR");
            IMR imr = new IMR(td_time, td_dirty, td_label, td_bool);
            double[][] td_repair_imr = imr.getTd_repair();
//            writeDataToCSV(td_repair_imr, file_path + "IMR.csv");
            cost_time = imr.getCost_time();
            detect_repair = AnomalyDetector.detect(td_repair_imr, p, beta);
            analysis = new Analysis(td_time, td_clean, td_repair_imr, td_bool, cost_time, detect_clean, detect_repair);
            rmseTable[dataset_idx][6] = analysis.getRMSEValue();
            timeTable[dataset_idx][6] = analysis.getCost_time();
//            recordFile(analysis.getRMSE() + ",", "RMSE");
//            recordFile(analysis.getPrecision() + ",", "Precision");
//            recordFile(analysis.getRecall() + ",", "Recall");
//            recordFile(analysis.getCost_time() + ",", "Time");
            System.gc();
            Runtime.getRuntime().gc();

            System.out.println("\nMTCSC");
            int dim = columnCnt;
            int T_mtcsc = p;
            Assist assist = new Assist();
            TimeSeriesN timeSeriesN = new TimeSeriesN(td_time, td_dirty, td_clean);
            double S = assist.getSpeedN(timeSeriesN, 1.0, dim);
            MTCSC mtcsc = new MTCSC(td_time, td_dirty, td_clean, S, T_mtcsc, dim);
            double[][] td_repair_mtcsc = mtcsc.getTd_repair();
//            writeDataToCSV(td_repair_mtcsc, file_path + "MTCSC.csv");
            cost_time = mtcsc.getCost_time();
            detect_repair = AnomalyDetector.detect(td_repair_mtcsc, p, beta);
            analysis = new Analysis(td_time, td_clean, td_repair_mtcsc, td_bool, cost_time, detect_clean, detect_repair);
            rmseTable[dataset_idx][7] = analysis.getRMSEValue();
            timeTable[dataset_idx][7] = analysis.getCost_time();
//            recordFile(analysis.getRMSE() + ",", "RMSE");
//            recordFile(analysis.getPrecision() + ",", "Precision");
//            recordFile(analysis.getRecall() + ",", "Recall");
//            recordFile(analysis.getCost_time() + ",", "Time");
            System.gc();
            Runtime.getRuntime().gc();
        }

        File resultsDir = new File("./results/whole_performance");
        if (!resultsDir.exists()) {
            resultsDir.mkdirs();
        }
        String rmseCsv = "./results/whole_performance/rmse.csv";
        String timeCsv = "./results/whole_performance/time.csv";
        writeCompareCsv(rmseCsv, datasets, methods, rmseTable, null);
        writeCompareCsv(timeCsv, datasets, methods, null, timeTable);
        System.out.println("Wrote RMSE CSV: " + rmseCsv);
        System.out.println("Wrote Time CSV: " + timeCsv);
    }

    /**
     * Same methods as {@link #get_data_repaired}, but z-score clean/dirty/domain
     * with clean-series mean/std after noise injection. Two trials, mean ± std.
     * ARM-only evaluation on z-scored series (clean/dirty/domain share clean μ,σ).
     * Two trials, mean ± std. RMSE is in normalized space.
     */
    public static void get_normalized_repaired(int rate) throws Exception {
        error_rate = rate;
        String[] datasets = DATASETS;
        String[] methods = {"ARM"};
        int nDatasets = datasets.length;
        final int nTrials = 2;
        final int baseSeed = 665;

        String[][] rmseCells = new String[nDatasets][1];
        String[][] timeCells = new String[nDatasets][1];

        File resultsDir = new File("./results/normalized_performance");
        if (!resultsDir.exists()) {
            resultsDir.mkdirs();
        }
        String rmseCsv = "./results/normalized_performance/rmse.csv";
        String timeCsv = "./results/normalized_performance/time.csv";

        try (BufferedWriter rmseOut = new BufferedWriter(new FileWriter(rmseCsv));
             BufferedWriter timeOut = new BufferedWriter(new FileWriter(timeCsv))) {
            rmseOut.write("dataset,ARM");
            rmseOut.newLine();
            timeOut.write("dataset,ARM");
            timeOut.newLine();

            System.out.println("\n===== Normalized ARM-only, " + nTrials + " trials (mean ± std) =====");
            for (int dataset_idx = 0; dataset_idx < nDatasets; dataset_idx++) {
                init(dataset_idx);
                seed = baseSeed;
                LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
                long[] td_time = loadData.getTd_time();
                double[][] origClean = sliceRows(loadData.getTd_clean(), 0, loadData.getTd_clean().length);
                KDTreeUtil kdTreeCompleteOrig = loadData.getKdTreeComplete();
                loadData.zscoreFromCleanSeries();
                double[][] td_clean = loadData.getTd_clean();
                KDTreeUtil kdTree = loadData.getKdTree();
                boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);
                int columnCnt = td_clean[0].length;

                System.out.println("\n===== dataset: " + datasets[dataset_idx] + " (z-scored) =====");
                System.out.println("  mu=" + Arrays.toString(loadData.getMean())
                        + "  sigma=" + Arrays.toString(loadData.getStd()));

                double[] rmses = new double[nTrials];
                double[] times = new double[nTrials];
                for (int t = 0; t < nTrials; t++) {
                    seed = baseSeed + t;
                    double[][] td_dirty = new AddNoise(origClean, error_rate, error_range, error_length,
                            eta, kdTreeCompleteOrig, seed).getTd_dirty();
                    loadData.applyZscore(td_dirty);
                    LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
                    ARMRepair armRepair = new ARMRepair(td_dirty, kdTree, td_time, columnCnt, k, p, eta, beta,
                            1, new VARUtil(p));
                    Analysis analysis = new Analysis(td_time, td_clean, armRepair.getTd_repaired(),
                            labelData.getTd_bool(), armRepair.getCost_time(), detect_clean,
                            AnomalyDetector.detect(armRepair.getTd_repaired(), p, beta));
                    rmses[t] = analysis.getRMSEValue();
                    times[t] = analysis.getCost_time();
                    System.out.println("  trial " + (t + 1) + "/" + nTrials
                            + " seed=" + seed
                            + " RMSE=" + String.format("%.6f", rmses[t])
                            + " Time=" + (long) times[t] + " ms"
                            + " Rounds=" + armRepair.getActual_rounds() + "/" + armRepair.getT());
                    gcQuiet();
                }
                rmseCells[dataset_idx][0] = formatMeanStd(rmses);
                timeCells[dataset_idx][0] = formatMeanStd(times);
                rmseOut.write(datasets[dataset_idx] + "," + rmseCells[dataset_idx][0]);
                rmseOut.newLine();
                timeOut.write(datasets[dataset_idx] + "," + timeCells[dataset_idx][0]);
                timeOut.newLine();
                rmseOut.flush();
                timeOut.flush();
                System.out.println("  RMSE: " + rmseCells[dataset_idx][0]);
                System.out.println("  Time (ms): " + timeCells[dataset_idx][0]);
            }
        }
        System.out.println("Wrote RMSE CSV: " + rmseCsv);
        System.out.println("Wrote Time CSV: " + timeCsv);
    }

    /**
     * Write comparison table CSV: header = methods, first column = datasets.
     * Pass either rmseTable or timeTable (the other null).
     */
    private static void writeCompareCsv(String path, String[] datasets, String[] methods,
                                        double[][] rmseTable, long[][] timeTable) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write("dataset");
            for (String method : methods) {
                bw.write(",");
                bw.write(method);
            }
            bw.newLine();
            for (int i = 0; i < datasets.length; i++) {
                bw.write(datasets[i]);
                for (int j = 0; j < methods.length; j++) {
                    bw.write(",");
                    if (rmseTable != null) {
                        bw.write(String.format("%.6f", rmseTable[i][j]));
                    } else {
                        bw.write(Long.toString(timeTable[i][j]));
                    }
                }
                bw.newLine();
            }
        }
    }

    private static void writeCellCsv(String path, String[] datasets, String[] methods,
                                     String[][] cells) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write("dataset");
            for (String method : methods) {
                bw.write(",");
                bw.write(method);
            }
            bw.newLine();
            for (int i = 0; i < datasets.length; i++) {
                bw.write(datasets[i]);
                for (int j = 0; j < methods.length; j++) {
                    bw.write(",");
                    bw.write(cells[i][j]);
                }
                bw.newLine();
            }
        }
    }

    // -------------------------------------------------------------------------
    // 8. Sensitivity experiments
    // -------------------------------------------------------------------------

    public static void whole_data_set(int dataset_idx) throws Exception {
        init(dataset_idx);
        recordBlankLines("traj_td-scale_whole\n");

        PreparedData d = prepareData(false, true);
        System.out.println("finish loading / labeling data");

        for (Analysis analysis : runRepairSuite(d, 5)) {
            recordAnalysis(analysis);
            gcQuiet();
        }
        recordBlankLines("\n\n");
    }

    public static void main_td_scale() throws Exception {
        System.out.println("--------------------");
        System.out.println("main td scale");
        System.out.println("--------------------");
        int td_len_base, td_len_step;
        String record;
        for (int dataset_idx = 4; dataset_idx < 5; dataset_idx++) {
            System.out.println("--------------------");
            System.out.println("in dataset " + dataset_idx);
            System.out.println("--------------------");
            if (dataset_idx == 0) {
                init(dataset_idx);
                td_len_base = 600 * 1000;
                td_len_step = 100 * 1000;
                record = "fuel_td-scale_0.6m,0.7m,0.8m,0.9m,1.0m,1.1m,1.2m,1.3m,1.4m,1.5m\n";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                td_len_base = 100 * 1000;
                td_len_step = 100 * 1000;
                record = "gps_td-scale_0.1m,0.2m,0.3m,0.4m,0.5m,0.6m,0.7m,0.8m,0.9m,1.0m\n";
            } else if (dataset_idx == 2) {
                init(dataset_idx);
                td_len_base = 600 * 1000;
                td_len_step = 100 * 1000;
                record = "road_td-scale_0.6m,0.7m,0.8m,0.9m,1.0m,1.1m,1.2m,1.3m,1.4m,1.5m\n";
            } else if (dataset_idx == 3)  {
                init(dataset_idx);
                td_len_base = 60 * 1000;
                td_len_step = 10 * 1000;
                record = "weather_td-scale_60k,70k,80k,90k,100k,120k,140k,160k,180k,200k\n";
            } else {
                init(dataset_idx);
                td_len_base = 2500;
                td_len_step = 2500;
                record = "traj_td-scale_2.5k,5k,7.5k,10k,12.5k,15k,17.5k,20k,22.5k,25k\n";
            }
            recordFile(record, "RMSE");
            recordFile(record, "Precision");
            recordFile(record, "Recall");
            recordFile(record, "Time");

            for (td_len = td_len_base; td_len <= td_len_base + td_len_step * 0; td_len += td_len_step) {
                System.out.println("td_len " + td_len);
                PreparedData d = prepareData(true, false);
                System.out.println("finish loading / noise / labeling");

                for (Analysis a : runRepairSuite(d, 5)) {
                    recordAnalysis(a);
                    gcQuiet();
                }
                recordBlankLines("\n\n");
            }
        }
    }


    public static void main_error_rate() throws Exception {
        System.out.println("--------------------");
        System.out.println("main error rate (road)");
        System.out.println("--------------------");
        // AddNoise uses per-mille: P(start error) = error_rate / 1000.
        // 10%..60% => 100, 200, 300, 400, 500, 600.
        int[] pctList = {10, 20, 30, 40, 50, 60};
        String[] methods = {"ARM"};
        final int dataset_idx = 2; // road

        init(dataset_idx);
        String record = "road_error-rate_10%,20%,30%,40%,50%,60% (AddNoise per-mille 100..600)\n";
        recordFile(record, "RMSE");
        recordFile(record, "Precision");
        recordFile(record, "Recall");
        recordFile(record, "Time");

        LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
        long[] td_time = loadData.getTd_time();
        double[][] td_clean = loadData.getTd_clean();
        double[][] domainData = loadData.getMd();
        KDTreeUtil kdTree = loadData.getKdTree();
        KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
        boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);
        System.out.println("finish load data, n=" + td_clean.length);
        System.out.println("AddNoise error_rate is per-mille; percents " + Arrays.toString(pctList)
                + " -> rates " + Arrays.toString(percentToPerMille(pctList)));

        int nRates = pctList.length;
        int nMethods = methods.length;
        double[][] rmseTable = new double[nRates][nMethods];
        long[][] timeTable = new long[nRates][nMethods];
        int[] rateList = new int[nRates];

        for (int r = 0; r < nRates; r++) {
            int pct = pctList[r];
            error_rate = pct * 10; // 10% -> 100 per-mille
            rateList[r] = pct;
            System.out.println("\n===== error rate " + pct + "% (per-mille=" + error_rate + ") =====");

            double[][] td_dirty = new AddNoise(td_clean, error_rate, error_range, error_length,
                    eta, kdTreeComplete, seed).getTd_dirty();
            LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
            PreparedData d = new PreparedData(td_time, null, td_clean, td_dirty, domainData,
                    labelData.getTd_label(), kdTree, kdTreeComplete, detect_clean, labelData.getTd_bool());

            Analysis[] results = runRepairSuite(d, 1);
            for (int j = 0; j < results.length; j++) {
                rmseTable[r][j] = results[j].getRMSEValue();
                timeTable[r][j] = results[j].getCost_time();
                recordAnalysis(results[j]);
                gcQuiet();
            }
            recordBlankLines("\n");
        }

        String outDir = "./results/error_rate/road";
        ensureDir(outDir);
        writeErrorRateCsv(outDir + "/rmse.csv", rateList, methods, rmseTable, null);
        writeErrorRateCsv(outDir + "/time.csv", rateList, methods, null, timeTable);
        System.out.println("Wrote " + outDir + "/rmse.csv and time.csv (error_rate column is percent)");
    }

    private static int[] percentToPerMille(int[] pctList) {
        int[] rates = new int[pctList.length];
        for (int i = 0; i < pctList.length; i++) {
            rates[i] = pctList[i] * 10;
        }
        return rates;
    }

    /** CSV for error-rate experiments: first column = error_rate, other columns = methods. */
    private static void writeErrorRateCsv(String path, int[] rateList, String[] methods,
                                          double[][] rmseTable, long[][] timeTable) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write("error_rate_pct");
            for (String method : methods) {
                bw.write(",");
                bw.write(method);
            }
            bw.newLine();
            for (int i = 0; i < rateList.length; i++) {
                bw.write(Integer.toString(rateList[i]));
                for (int j = 0; j < methods.length; j++) {
                    bw.write(",");
                    if (rmseTable != null) {
                        bw.write(String.format("%.6f", rmseTable[i][j]));
                    } else {
                        bw.write(Long.toString(timeTable[i][j]));
                    }
                }
                bw.newLine();
            }
        }
    }

    public static void main_error_range() throws Exception {
        System.out.println("--------------------");
        System.out.println("main error range");
        System.out.println("--------------------");
        double error_range_base, error_range_step;
        String record;
        for (int dataset_idx = 0; dataset_idx < 4; dataset_idx++) {
            System.out.println("--------------------");
            System.out.println("in dataset " + dataset_idx);
            System.out.println("--------------------");
            if (dataset_idx == 0) {
                init(dataset_idx);
                error_range_base = 0.5;
                error_range_step = 0.5;
                record = "fuel_error-range_0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0\n";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                error_range_base = 0.2;
                error_range_step = 0.2;
                record = "gps_error-range_0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0\n";
            } else if (dataset_idx == 2) {
                init(dataset_idx);
                error_range_base = 0.5;
                error_range_step = 0.5;
                record = "road_error-range_0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0\n";
            } else {
                init(dataset_idx);
                error_range_base = 0.5;
                error_range_step = 0.5;
                record = "weather_error-range_0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0\n";
            }
            recordBlankLines(record);

            LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
            long[] td_time = loadData.getTd_time();
            double[][] td_clean = loadData.getTd_clean();
            double[][] domainData = loadData.getMd();
            KDTreeUtil kdTree = loadData.getKdTree();
            KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
            boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

            for (error_range = error_range_base;
                 error_range <= error_range_base + error_range_step * 9;
                 error_range += error_range_step) {
                double[][] td_dirty = new AddNoise(td_clean, error_rate, error_range, error_length,
                        eta, kdTreeComplete, seed).getTd_dirty();
                LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
                PreparedData d = new PreparedData(td_time, null, td_clean, td_dirty, domainData,
                        labelData.getTd_label(), kdTree, kdTreeComplete, detect_clean, labelData.getTd_bool());
                for (Analysis a : runRepairSuite(d, 5)) {
                    recordAnalysis(a);
                    gcQuiet();
                }
                recordBlankLines("\n");
            }
        }
    }

    public static void main_error_length() throws Exception {
        System.out.println("--------------------");
        System.out.println("main error length");
        System.out.println("--------------------");
        for (int dataset_idx = 0; dataset_idx < 4; dataset_idx++) {
            System.out.println("--------------------");
            System.out.println("in dataset " + dataset_idx);
            System.out.println("--------------------");
            init(dataset_idx);
            String[] lengthLabels = {
                    "fuel_error-length_1,2,3,4,5,6,7,8,9,10\n",
                    "gps_error-length_1,2,3,4,5,6,7,8,9,10\n",
                    "road_error-length_1,2,3,4,5,6,7,8,9,10\n",
                    "weather_error-length_1,2,3,4,5,6,7,8,9,10\n"
            };
            recordBlankLines(lengthLabels[dataset_idx]);

            LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
            long[] td_time = loadData.getTd_time();
            double[][] td_clean = loadData.getTd_clean();
            double[][] domainData = loadData.getMd();
            KDTreeUtil kdTree = loadData.getKdTree();
            KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
            boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

            for (error_length = 1; error_length <= 10; error_length++) {
                double[][] td_dirty = new AddNoise(td_clean, error_rate, error_range, error_length,
                        eta, kdTreeComplete, seed).getTd_dirty();
                LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
                PreparedData d = new PreparedData(td_time, null, td_clean, td_dirty, domainData,
                        labelData.getTd_label(), kdTree, kdTreeComplete, detect_clean, labelData.getTd_bool());
                for (Analysis a : runRepairSuite(d, 5)) {
                    recordAnalysis(a);
                    gcQuiet();
                }
                recordBlankLines("\n");
            }
        }
    }

    public static void main_parameters(boolean k_bool, boolean p_bool, boolean eta_bool, boolean beta_bool)
            throws Exception {
        System.out.println("--------------------");
        System.out.println("main parameters");
        System.out.println("--------------------");
        String[] prefixes = {"fuel_", "gps_", "road_", "weather_"};
        for (int dataset_idx = 0; dataset_idx < 4; dataset_idx++) {
            System.out.println("--------------------");
            System.out.println("in dataset " + dataset_idx);
            System.out.println("--------------------");
            init(dataset_idx);
            String record = prefixes[dataset_idx];

            PreparedData d = prepareData(true, false);
            System.out.println("finish load / noise / labeling");

            if (k_bool) {
                init(dataset_idx);
                recordBlankLines(record + "k_1,2,3,4,5\n");
                for (k = 1; k <= 5; k++) {
                    Analysis a = armRepair(d.kdTree, d.td_time, d.td_clean, d.td_dirty,
                            d.default_bool, d.detect_clean);
                    recordFile(a.getRMSE() + ",\n", "RMSE");
                    recordFile(a.getPrecision() + ",\n", "Precision");
                    recordFile(a.getRecall() + ",\n", "Recall");
                    recordFile(a.getCost_time() + ",\n", "Time");
                    gcQuiet();
                }
            }

            if (eta_bool) {
                init(dataset_idx);
                recordBlankLines(record + "eta_0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0\n");
                for (eta = 0.2; eta <= 2.0; eta += 0.2) {
                    Analysis a = armRepair(d.kdTree, d.td_time, d.td_clean, d.td_dirty,
                            d.default_bool, d.detect_clean);
                    recordFile(a.getRMSE() + ",\n", "RMSE");
                    recordFile(a.getPrecision() + ",\n", "Precision");
                    recordFile(a.getRecall() + ",\n", "Recall");
                    recordFile(a.getCost_time() + ",\n", "Time");
                    gcQuiet();
                }
            }

            if (p_bool) {
                init(dataset_idx);
                recordBlankLines(record + "p_5,10,15,20,25\n");
                for (p = 5; p <= 25; p += 5) {
                    for (Analysis a : runRepairSuite(d, 5)) {
                        recordAnalysis(a);
                        gcQuiet();
                    }
                    recordBlankLines("\n");
                }
            }

            if (beta_bool) {
                init(dataset_idx);
                String parameter_type = "beta_0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5\n";
                recordFile(record + parameter_type, "Precision");
                recordFile(record + parameter_type, "Recall");
                recordFile(record + parameter_type, "RMSE");
                recordFile(record + parameter_type, "Time");
                for (beta = 0.25; beta <= 2.5; beta += 0.25) {
                    for (Analysis a : runRepairSuite(d, 5)) {
                        recordFile(a.getPrecision() + ",", "Precision");
                        recordFile(a.getRecall() + ",", "Recall");
                        recordFile(a.getRMSE() + ",", "RMSE");
                        recordFile(a.getCost_time() + ",", "Time");
                        gcQuiet();
                    }
                    recordFile("\n", "Precision");
                    recordFile("\n", "Recall");
                    recordFile("\n", "RMSE");
                    recordFile("\n", "Time");
                }
            }
        }
    }
}
