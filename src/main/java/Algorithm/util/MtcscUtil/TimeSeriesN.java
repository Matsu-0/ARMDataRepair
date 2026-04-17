package Algorithm.util.MtcscUtil;

import java.util.ArrayList;

public class TimeSeriesN {
    private ArrayList<TimePointN> Timeseries;

	private int length;

	private int dim;

	public TimeSeriesN(ArrayList<TimePointN> timeseries) {
		setTimeseries(timeseries);
	}

	public TimeSeriesN() {
		setTimeseries(new ArrayList<TimePointN>());
	}

	public TimeSeriesN(long[] td_time, double[][] td_dirty, double[][] td_clean) {
		if (td_dirty.length == 0 || td_dirty[0].length == 0) {
			setTimeseries(new ArrayList<TimePointN>());
			return;
		}
		this.length = td_dirty.length;
		this.dim = td_dirty[0].length;
		this.Timeseries = new ArrayList<TimePointN>();
		TimePointN timePointN;

		for (int i = 0; i < this.length; i++) {
			long timestamp;
			ArrayList<Double> value = new ArrayList<Double>();
			ArrayList<Double> truth = new ArrayList<Double>();

			double[] dirty_point = td_dirty[i];
			double[] clean_point = td_clean[i];
			timestamp = td_time[i];
			for (int j = 0; j < this.dim; j++) {
				value.add(dirty_point[j]);
				truth.add(clean_point[j]);
			}
			timePointN = new TimePointN(dim, timestamp, value, truth);
			this.Timeseries.add(timePointN);
		}
	}

	public ArrayList<TimePointN> getTimeseries() {
		return this.Timeseries;
	}

	public void setTimeseries(ArrayList<TimePointN> timeseries) {
		this.Timeseries = timeseries;
	}

    public void addPoint(TimePointN tp) {
		this.Timeseries.add(tp);
	}

	public int getLength() {
		return this.Timeseries.size();
	}

    public void removePoint(int index) {
		this.Timeseries.remove(index);
    }

	public double[][] getRepairedSeries() {
		double[][] repairedSeries = new double[this.Timeseries.size()][this.dim];
		for (int i = 0; i < this.Timeseries.size(); i++) {
			TimePointN timePointN = this.Timeseries.get(i);
			ArrayList<Double> modified = timePointN.getModify();
			double[] repaired = new double[this.dim];
			for (int j = 0; j < this.dim; j++) {
				repaired[j] = modified.get(j);
			}
			repairedSeries[i] = repaired;
		}
		return repairedSeries;
	}
	
}
