package Algorithm;

import Algorithm.util.IMRUtil;

public class IMR {
    private final long[] td_time;
    private final double[][] td_dirty;
    private final double[][] td_repair;
    private final double[][] td_label;
    private final boolean[] td_bool;
    private final long cost_time;

    public IMR(long[] td_time, double[][] td_dirty, double[][] td_label, boolean[] td_bool) {
        this.td_time = td_time;
        this.td_dirty = td_dirty;
        this.td_label = td_label;
        this.td_bool = td_bool;
        this.td_repair = new double[td_dirty.length][td_dirty[0].length];

        long startTime = System.currentTimeMillis();
        this.repair();
        long endTime = System.currentTimeMillis();
        this.cost_time = endTime - startTime;
        System.out.println("IMR time cost:" + cost_time + "ms");
    }

    public double[][] getTd_repair() {
        return td_repair;
    }

    public long getCost_time() {
        return cost_time;
    }

    private void repair() {
        IMRUtil imrUtil;
        double[] td_dirty_col = new double[td_dirty.length], td_label_col = new double[td_label.length];
        for (int col = 0, row; col < td_dirty[0].length; col++) {
            for (row = 0; row < td_dirty.length; row++) {  // input
                td_dirty_col[row] = td_dirty[row][col];
                td_label_col[row] = td_label[row][col];
            }

            imrUtil = new IMRUtil(td_time, td_dirty_col, td_label_col, td_bool, 3, 1.5, 10000);
            imrUtil.repair();
            td_dirty_col = imrUtil.getRepaired();

            for (row = 0; row < td_dirty.length; row++) // output
                td_repair[row][col] = td_dirty_col[row];
        }
    }
}
