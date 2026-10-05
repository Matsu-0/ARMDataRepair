package Algorithm;

import Algorithm.util.TimeSeriesPredictor;
import Algorithm.util.VARUtil;

import java.util.ArrayList;

/**
 * Baseline: no domain constraints. Fit a predictor on the dirty series, detect
 * points whose residual vs prediction exceeds beta (same rule as
 * {@link AnomalyDetector}), and repair each with the predicted value.
 *
 * Call pattern matches {@link ARMRepair}: construct to run, then
 * {@link #getTd_repaired()} / {@link #getCost_time()}.
 */
public class ModelOnlyRepair {
    private final double[][] td;
    private final int columnCnt;
    private final int n;
    private final int p;
    private final double beta;
    private final TimeSeriesPredictor prediction_model;
    private final double[][] td_repaired;
    private final long cost_time;

    public ModelOnlyRepair(double[][] td, long[] td_time, int columnCnt, int p, double beta) {
        this(td, td_time, columnCnt, p, beta, new VARUtil(p));
    }

    public ModelOnlyRepair(double[][] td, long[] td_time, int columnCnt, int p, double beta,
                           TimeSeriesPredictor predictor) {
        this.td = td;
        this.columnCnt = columnCnt;
        this.n = td.length;
        this.p = p;
        this.beta = beta;
        this.prediction_model = predictor != null ? predictor : new VARUtil(p);
        this.td_repaired = new double[n][columnCnt];
        long start = System.nanoTime();
        this.repair();
        this.cost_time = (System.nanoTime() - start) / 1_000_000L;
        System.out.println("ModelOnlyRepair time cost:" + cost_time + "ms");
    }

    private void repair() {
        ArrayList<ArrayList<Double>> samples = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            ArrayList<Double> sample = new ArrayList<>(columnCnt);
            for (int c = 0; c < columnCnt; c++) {
                sample.add(td[i][c]);
            }
            samples.add(sample);
            td_repaired[i] = td[i].clone();
        }
        prediction_model.fit(samples);

        int repairedCnt = 0;
        for (int i = 0; i < p; i++) {
            td_repaired[i] = td[i].clone();
        }
        for (int i = p; i < n; i++) {
            double[][] window = getWindow(td, i, p);
            double[] predicted = arrayToList(prediction_model.predict(window));
            if (residualDistance(predicted, td[i]) > beta) {
                td_repaired[i] = predicted;
                repairedCnt++;
            } else {
                td_repaired[i] = td[i].clone();
            }
        }
        System.out.println("ModelOnlyRepair repaired " + repairedCnt + " / " + n + " points (beta)");
    }

    private double[][] getWindow(double[][] data, int i, int window) {
        double[][] W = new double[window][columnCnt];
        System.arraycopy(data, i - window, W, 0, window);
        return W;
    }

    private static double residualDistance(double[] predicted, double[] observed) {
        double distance = 0d;
        for (int pos = 0; pos < predicted.length; pos++) {
            double temp = predicted[pos] - observed[pos];
            distance += temp * temp;
        }
        return Math.sqrt(distance);
    }

    private static double[] arrayToList(ArrayList<Double> arrayList) {
        double[] list = new double[arrayList.size()];
        for (int i = 0; i < arrayList.size(); i++) {
            list[i] = arrayList.get(i);
        }
        return list;
    }

    public double[][] getTd_repaired() {
        return td_repaired;
    }

    public double[][] getTd_repair() {
        return td_repaired;
    }

    public long getCost_time() {
        return cost_time;
    }
}
