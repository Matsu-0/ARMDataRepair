package Algorithm;

public class EditingRuleRepair {
    private final long[] td_time;
    private final double[][] td_dirty;
    private final double[][] td_repair;
    private final double[][] md;

    private final double[] tolerance;

    private final long cost_time;

    public EditingRuleRepair(long[] td_time, double[][] td_dirty, double[][] md) {
        this.td_time = td_time;
        this.td_dirty = td_dirty;
        this.td_repair = new double[td_dirty.length][td_dirty[0].length];
        this.md = md;

        this.tolerance = new double[td_dirty[0].length];

        this.preprocess();
        long startTime = System.currentTimeMillis();
        this.repair();
        long endTime = System.currentTimeMillis();
        this.cost_time = endTime - startTime;
        System.out.println("EditingRule Time cost:" + (endTime - startTime) + "ms");
    }

    private void preprocess() {
        for (int i = 0; i < td_dirty[0].length; i++) {
            this.tolerance[i] = Math.abs(this.md[0][i] - this.md[1][i]);
        }
    }

    public double[][] getTd_repair() {
        return td_repair;
    }

    public long getCost_time() {
        return cost_time;
    }

    private void repair() {
        boolean repaired;
        double[] td_row = new double[td_dirty[0].length];
        for (int row = 0; row < td_dirty.length; row++) {
            System.arraycopy(td_dirty[row], 0, td_row, 0, td_dirty[0].length);

            repaired = false;
            for (int diff = 1; diff < td_dirty[0].length; diff++) {
                repaired = repairWithEditRule(td_row, row, diff);
                if (repaired) {
                    break;
                }
            }
            if (!repaired) {
                System.arraycopy(td_row, 0, td_repair[row], 0, td_dirty[0].length);
            }
        }
    }

    private boolean repairWithEditRule(double[] td_row, int row, int diff) {
        double[] md_row = new double[md[0].length];
        for (int r = 0; r < md.length; r++) {
            System.arraycopy(md[r], 0, md_row, 0, md[0].length);
            if (this.editRule(td_row, md_row, diff)) {
                System.arraycopy(md_row, 0, td_repair[row], 0, td_dirty[0].length);
                return true;
            }
        }
        return false;
    }

    private boolean editRule(double[] td_row, double[] md_row, int diff) {
        int same_cnt = 0, column_num = td_dirty[0].length;
        for (int i = 0; i < column_num; i++) {
            if (Math.abs(td_row[i] - md_row[i]) < this.tolerance[i]) {
                same_cnt++;
            }
        }
        return same_cnt >= column_num - diff;
    }
}
