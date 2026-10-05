package Algorithm;

import Algorithm.util.KDTreeUtil;

/**
 * Baseline: no prediction model. Detect points that violate domain constraints
 * (eta), and repair each with its nearest neighbor in the domain-constraint set.
 *
 * Call pattern matches {@link ARMRepair}: construct to run, then
 * {@link #getTd_repaired()} / {@link #getCost_time()}.
 */
public class DomainNNRepair {
    private final double[][] td;
    private final KDTreeUtil kdTreeUtil;
    private final int columnCnt;
    private final int n;
    private final double eta;
    private final double[] std;
    private final double[][] td_repaired;
    private final long cost_time;

    public DomainNNRepair(double[][] td, KDTreeUtil kdTreeUtil, long[] td_time, int columnCnt, double eta) {
        this.td = td;
        this.kdTreeUtil = kdTreeUtil;
        this.columnCnt = columnCnt;
        this.n = td.length;
        this.eta = eta;
        this.std = computeStd();
        this.td_repaired = new double[n][columnCnt];
        long start = System.nanoTime();
        this.repair();
        this.cost_time = (System.nanoTime() - start) / 1_000_000L;
        System.out.println("DomainNNRepair time cost:" + cost_time + "ms");
    }

    private void repair() {
        int repairedCnt = 0;
        for (int i = 0; i < n; i++) {
            if (!checkConsistency(td[i])) {
                double[] nn = kdTreeUtil.nearestNeighbor(td[i]);
                td_repaired[i] = nn.clone();
                repairedCnt++;
            } else {
                td_repaired[i] = td[i].clone();
            }
        }
        System.out.println("DomainNNRepair repaired " + repairedCnt + " / " + n + " points (eta)");
    }

    private boolean checkConsistency(double[] tuple) {
        double[] nn = kdTreeUtil.nearestNeighbor(tuple);
        return delta(tuple, nn) <= eta;
    }

    /** Same signed, std-normalized delta as {@link ARMRepair}. */
    private double delta(double[] t_tuple, double[] m_tuple) {
        double distance = 0d;
        for (int pos = 0; pos < columnCnt; pos++) {
            distance += (t_tuple[pos] - m_tuple[pos]) / std[pos];
        }
        return distance;
    }

    private double[] computeStd() {
        double[] result = new double[columnCnt];
        for (int c = 0; c < columnCnt; c++) {
            double sum = 0d;
            int cnt = 0;
            for (int i = 0; i < n; i++) {
                if (!Double.isNaN(td[i][c])) {
                    sum += td[i][c];
                    cnt++;
                }
            }
            double mean = cnt == 0 ? 0d : sum / cnt;
            double var = 0d;
            for (int i = 0; i < n; i++) {
                if (!Double.isNaN(td[i][c])) {
                    double d = td[i][c] - mean;
                    var += d * d;
                }
            }
            result[c] = cnt == 0 ? 1d : Math.sqrt(var / cnt);
            if (result[c] == 0d) {
                result[c] = 1d;
            }
        }
        return result;
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
