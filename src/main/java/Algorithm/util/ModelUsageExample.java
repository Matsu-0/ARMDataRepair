package Algorithm.util;

import Algorithm.AnomalyDetector;
import Algorithm.MasterDetector;
import Algorithm.util.KDTreeUtil;

/**
 * Example of swapping prediction models.
 * Shows how to use different predictors (VAR, DLinear, etc.).
 */
public class ModelUsageExample {
    
    /**
     * Example 1: Use default VAR model (backward compatible).
     */
    public static void example1_UseVAR(double[][] td, KDTreeUtil kdTree, long[] td_time, 
                                       int columnCnt, int k, int p, double eta) {
        // Option 1: use default constructor (VAR is used automatically)
        MasterDetector detector1 = new MasterDetector(td, kdTree, td_time, columnCnt, k, p, eta);
        
        // Option 2: explicitly pass VAR model
        VARUtil varModel = new VARUtil(columnCnt);
        MasterDetector detector2 = new MasterDetector(td, kdTree, td_time, columnCnt, k, p, eta, varModel);
    }
    
    /**
     * Example 2: Use DLinear model (Python version).
     */
    public static void example2_UseDLinear(double[][] td, KDTreeUtil kdTree, long[] td_time, 
                                           int columnCnt, int k, int p, double eta) {
        // first check Python environment
        String pythonScriptPath = "./python/dlinear_model.py";
        if (!DLinearUtil.checkPythonEnvironment(pythonScriptPath)) {
            System.err.println("Warning: Python environment not ready; DLinear may not work properly");
        }
        
        // create DLinear model (using Python script)
        DLinearUtil dlinearModel = new DLinearUtil(
            p,                          // window size
            columnCnt,                  // number of variables
            pythonScriptPath,           // path to Python script
            "./models"                  // model save directory
        );
        
        // create MasterDetector with DLinear model
        MasterDetector detector = new MasterDetector(td, kdTree, td_time, columnCnt, k, p, eta, dlinearModel);
        
        // get repair result
        double[][] repaired = detector.getTd_repaired();
    }
    
    /**
     * Example 2b: Use DLinear model with custom parameters.
     */
    public static void example2b_UseDLinearCustom(double[][] td, KDTreeUtil kdTree, long[] td_time, 
                                                   int columnCnt, int k, int p, double eta) {
        // create DLinear model with custom training parameters
        DLinearUtil dlinearModel = new DLinearUtil(
            p,                          // window size
            columnCnt,                  // number of variables
            "./python/dlinear_model.py", // path to Python script
            "./models",                 // model save directory
            true,                       // individual mode (each variable modeled separately)
            20,                         // training epochs
            0.0005                      // learning rate
        );
        
        // create MasterDetector with DLinear model
        MasterDetector detector = new MasterDetector(td, kdTree, td_time, columnCnt, k, p, eta, dlinearModel);
        
        // get repair result
        double[][] repaired = detector.getTd_repaired();
    }
    
    /**
     * Example 3: Use DLinear in anomaly detection.
     */
    public static void example3_AnomalyDetectionWithDLinear(double[][] td_repaired, int p, double beta) {
        // create DLinear model (Python version)
        int columnCnt = td_repaired[0].length;
        DLinearUtil dlinearModel = new DLinearUtil(
            p, 
            columnCnt,
            "./python/dlinear_model.py",
            "./models"
        );
        
        // run anomaly detection with DLinear
        boolean[] anomalies = AnomalyDetector.detect(td_repaired, p, beta, dlinearModel);
    }
    
    /**
     * Example 4: Use absolute paths (when relative paths do not work).
     */
    public static void example4_UseAbsolutePath(double[][] td, KDTreeUtil kdTree, long[] td_time, 
                                                 int columnCnt, int k, int p, double eta) {
        // use absolute paths for Python script and model directory
        String absolutePythonPath = "/absolute/path/to/python/dlinear_model.py";
        String absoluteModelDir = "/absolute/path/to/models";
        
        DLinearUtil dlinearModel = new DLinearUtil(
            p,
            columnCnt,
            absolutePythonPath,
            absoluteModelDir
        );
        
        MasterDetector detector = new MasterDetector(td, kdTree, td_time, columnCnt, k, p, eta, dlinearModel);
    }
    
    /**
     * Example 5: How to add a new model (e.g. LSTM).
     * 
     * Steps:
     * 1. Create a new class implementing TimeSeriesPredictor
     * 2. Implement fit() and predict()
     * 3. Use it in MasterDetector
     * 
     * Example code structure:
     * 
     * public class LSTMUtil implements TimeSeriesPredictor {
     *     private int p;
     *     private int columnCnt;
     *     // LSTM-related fields...
     *     
     *     public LSTMUtil(int p, int columnCnt) {
     *         this.p = p;
     *         this.columnCnt = columnCnt;
     *     }
     *     
     *     @Override
     *     public void fit(ArrayList<ArrayList<Double>> data) {
     *         // train LSTM model
     *     }
     *     
     *     @Override
     *     public ArrayList<Double> predict(double[][] window) {
     *         // predict with LSTM
     *     }
     *     
     *     @Override
     *     public int getWindowSize() {
     *         return p;
     *     }
     * }
     */
}
