import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

public class Analysis {
    private final double[][] td_clean;
    private final double[][] td_repair;
    private final boolean[] td_bool;

    private final boolean[] detect_clean;
    private final boolean[] detect_repair;

    private double RMSE;
    private double F1score;
    private double precision;
    private double recall;
    private final long cost_time;

    public Analysis(long[] td_time, double[][] td_clean, double[][] td_repair, boolean[] td_bool, long cost_time, boolean[] detect_clean, boolean[] detect_repair) {
        this.td_clean = td_clean;
        this.td_repair = td_repair;
        this.td_bool = td_bool;
        this.cost_time = cost_time;

        this.detect_clean = detect_clean;
        this.detect_repair = detect_repair;

        this.analysisRMSE();
        this.analysisF1score();
    }

    private void analysisF1score() {
        int tp = 0, fp = 0, fn = 0;

        for (int i = 0; i < detect_repair.length; i++) {
            if (detect_repair[i] && detect_clean[i]) {
                tp++;
            } else if (detect_repair[i] && !detect_clean[i]) {
                fp++;
            } else if (!detect_repair[i] && detect_clean[i]) {
                fn++;
            }
        }
        tp++;
        fp++;
        fn++;

        this.precision = tp / (double) (tp + fp);
        this.recall = tp / (double) (tp + fn);
        this.F1score = (2 * precision * recall) / (precision + recall);
    }

    private void analysisRMSE() {
        int td_len = td_repair.length, td_col_len = td_repair[0].length, label_num = 0;
        for (int row = 0; row < td_len; row++) {
            if (td_bool[row]) {
                label_num++;
                continue;
            }
            for (int col = 0; col < td_col_len; col++) {
                this.RMSE += Math.pow(td_clean[row][col] - td_repair[row][col], 2);
            }
        }
        this.RMSE = Math.sqrt(this.RMSE / (td_len - label_num) / td_col_len);
    }

    public String getRMSE() {
        return String.format("%.3f", this.RMSE);
    }
    
    public double getRMSEValue() {
        return this.RMSE;
    }

    public String getPrecision() {
        return String.format("%.3f", this.precision);
    }

    public String getRecall() {
        return String.format("%.3f", this.recall);
    }

    public String getF1score() {
        return String.format("%.3f", this.F1score);
    }

    public long getCost_time() {
        return this.cost_time;
    }
}
