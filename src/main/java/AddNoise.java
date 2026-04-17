import Algorithm.util.KDTreeUtil;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Random;

public class AddNoise {
    private final double[][] td_clean;
    private final double[][] td_dirty;

    private final int err_rate;  // error rate rate/1000
    private final double err_range;  // error range
    private final int err_length;  // error rate rate/1000

    private final double eta;
    private final KDTreeUtil kdTree;

    private final Random random;

    public AddNoise(double[][] td_clean, int rate, double range, int length, double eta, KDTreeUtil kdTree, int seed) {
        this.td_clean = td_clean;
        this.td_dirty = new double[td_clean.length][td_clean[0].length];

        this.err_rate = rate;
        this.err_range = range;
        this.err_length = length;

        this.eta = eta;
        this.kdTree = kdTree;

        this.random = new Random(seed);

        addNoise();
    }

    private void addNoise() {
        int err_flag = 0, error_fault_num = 0, i;
        double err_range_random = 0.0, value_dirty, dist;
        for (int row = 0; row < td_clean.length; row++) {
            if (random.nextInt(1000) < this.err_rate) {
                err_flag = random.nextInt(err_length * 2 - 1) + 1;  // mean = error length
                err_range_random = random.nextGaussian();
            }


            if (err_flag > 0) {
                --err_flag;
                dist = kdTree.nearestNeighborDistance(td_clean[row]);
                for (i = 0; i < 100 && dist < eta; i++) {
                    for (int col = 0; col < td_clean[0].length; col++) {
                        value_dirty = td_clean[row][col] + err_range_random * this.err_range;
                        BigDecimal b = new BigDecimal(value_dirty);
                        td_dirty[row][col] = b.setScale(5, RoundingMode.HALF_UP).doubleValue();
                    }
                    dist = kdTree.nearestNeighborDistance(td_dirty[row]);
                    if (dist < eta) err_range_random = random.nextGaussian();
                }
                if (i == 100)
                    error_fault_num += 1;
            } else {
                System.arraycopy(td_clean[row], 0, td_dirty[row], 0, td_clean[0].length);
            }
        }
        if (error_fault_num > 0)
            System.out.println("Error Inject Fault: The error range may be too small, error fault num: " + error_fault_num);
    }

    public double[][] getTd_dirty() {
        return td_dirty;
    }
}
