package Algorithm;

import Algorithm.util.TimeSeriesPredictor;
import Algorithm.util.VARUtil;

import java.util.ArrayList;

public class AnomalyDetector {
    public static boolean[] detect(double[][] td_repaired, int p, double beta) {
        return detect(td_repaired, p, beta, new VARUtil(p));
    }
    
    public static boolean[] detect(double[][] td_repaired, int p, double beta, TimeSeriesPredictor predictor) {
        int columnCnt = td_repaired[0].length;
        TimeSeriesPredictor prediction_model = predictor;
        prediction_model.fit(listToArray(td_repaired));
        boolean[] anomalies = new boolean[td_repaired.length];
        for (int i = 0; i < p; i++) {
            anomalies[i] = Boolean.FALSE;
        }
        for (int i = p; i < td_repaired.length; i++) {
            double[] tuple = td_repaired[i];
            double[][] window = getWindow(td_repaired, i, p, columnCnt);
            assert window != null;
            ArrayList<Double> predict_result = prediction_model.predict(window);
            if (delta(arrayToList(predict_result), tuple) > beta) {
                anomalies[i] = Boolean.TRUE;
            } else {
                anomalies[i] = Boolean.FALSE;
            }
        }
        return anomalies;
    }

    private static double delta(double[] t_tuple, double[] m_tuple) {
        double distance = 0d;
        for (int pos = 0; pos < t_tuple.length; pos++) {
            double temp = t_tuple[pos] - m_tuple[pos];
            distance += temp * temp;
        }
        distance = Math.sqrt(distance);
        return distance;
    }

    private static double[][] getWindow(double[][] data, int i, int p, int columnCnt) {
        if (i < p) {
            System.out.println("ERROR: i must be greater than p.");
            return null;
        }
        double[][] W = new double[p][columnCnt];
        System.arraycopy(data, i - p, W, 0, p);
        return W;
    }

    private static ArrayList<ArrayList<Double>> listToArray(double[][] list) {
        ArrayList<ArrayList<Double>> array = new ArrayList<>();
        for (double[] tuple : list) {
            ArrayList<Double> t = new ArrayList<>();
            for (double value : tuple) {
                t.add(value);
            }
            array.add(t);
        }
        return array;
    }

    private static double[] arrayToList(ArrayList<Double> arrayList) {
        double[] doubleArray = new double[arrayList.size()];
        for (int i = 0; i < arrayList.size(); i++) {
            doubleArray[i] = arrayList.get(i);
        }
        return doubleArray;
    }
}
