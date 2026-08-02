package Algorithm.util;

import java.util.ArrayList;

/**
 * Time series prediction model interface.
 * Supports pluggable prediction models (VAR, DLinear, LSTM, etc.).
 */
public interface TimeSeriesPredictor {
    
    /**
     * Train the model (default one-step ahead).
     * @param data training data; each ArrayList&lt;Double&gt; represents multi-variate data at one time step
     */
    void fit(ArrayList<ArrayList<Double>> data);
    
    /**
     * Predict based on sliding window.
     * @param window sliding window data; window[i] is the multi-variate data at the i-th time step
     * @return prediction for the next time step (multi-variate)
     */
    ArrayList<Double> predict(double[][] window);

    /**
     * Direct multi-step prediction: from {@code window}, predict the next {@code horizon} steps at once.
     * @return preds[step][dim] with length {@code horizon}
     */
    double[][] predictHorizon(double[][] window, int horizon);
    
    /**
     * Get the window size required by the model.
     * @return window size (p)
     */
    int getWindowSize();
}
