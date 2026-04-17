package Algorithm;

import Algorithm.util.MtcscUtil.Assist;
import Algorithm.util.MtcscUtil.MTCSC_N;
import Algorithm.util.MtcscUtil.TimeSeriesN;

public class MTCSC {
    private final long[] td_time;
    private final double[][] td_dirty;
    private final double[][] td_clean;
    private final double[][] td_repair;
    private final long cost_time;

    public MTCSC(long[] td_time, double[][] td_dirty, double[][] td_clean, double S, int T, int dim) throws Exception {
        this.td_time = td_time;
        this.td_dirty = td_dirty;
        this.td_clean = td_clean;

        TimeSeriesN timeSeriesN = new TimeSeriesN(td_time, td_dirty, td_clean);

        long startTime = System.currentTimeMillis();
        MTCSC_N tp = new MTCSC_N(timeSeriesN, S, T, dim);
        TimeSeriesN resultSeries = tp.mainScreen();
        this.td_repair = resultSeries.getRepairedSeries();
        long endTime = System.currentTimeMillis();
        this.cost_time = endTime - startTime;
        System.out.println("MTCSC time cost:" + cost_time + "ms");

        Assist assist = new Assist();

        double rms_MyN = assist.RMSN(resultSeries, dim);
        double cost_MyN = assist.CostNN(resultSeries, dim);
        int num_MyN = assist.pointNumN(resultSeries, dim);
        System.out.println(rms_MyN);
        System.out.println(cost_MyN);
        System.out.println(num_MyN);
    }

    public double[][] getTd_repair() {
        return td_repair;
    }

    public long getCost_time() {
        return cost_time;
    }
}
