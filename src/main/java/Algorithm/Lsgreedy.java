package Algorithm;

import Algorithm.util.LsgreedyUtil;

public class Lsgreedy {
    private final long[] td_time;
    private final double[][] td_dirty;
    private final double[][] td_repair;
    private final long cost_time;

    public Lsgreedy(long[] td_time, double[][] td_dirty) throws Exception {
        this.td_time = td_time;
        this.td_dirty = td_dirty;
        this.td_repair = new double[td_dirty.length][td_dirty[0].length];

        long startTime = System.currentTimeMillis();
        this.repair();
        long endTime = System.currentTimeMillis();
        this.cost_time = endTime - startTime;
        System.out.println("Lsgreedy time cost:" + cost_time + "ms");
    }

    public double[][] getTd_repair() {
        return td_repair;
    }

    public long getCost_time() {
        return cost_time;
    }

    private void repair() throws Exception {
        LsgreedyUtil lsgreedyUtil;
        double[] td_col = new double[td_dirty.length];
        for (int col = 0, row; col < td_dirty[0].length; col++) {
            for (row = 0; row < td_dirty.length; row++)  // input
                td_col[row] = td_dirty[row][col];

            lsgreedyUtil = new LsgreedyUtil(td_time, td_col);
            lsgreedyUtil.repair();
            td_col = lsgreedyUtil.getRepaired();

            for (row = 0; row < td_dirty.length; row++) { // output
                if (!Double.isNaN(td_col[row])) {
                    td_repair[row][col] = td_col[row];
                } else {
                    td_repair[row][col] = td_col[row - 1];
                }
            }
        }
    }
}
