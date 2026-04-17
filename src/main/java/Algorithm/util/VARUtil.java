package Algorithm.util;
import java.util.ArrayList;

public class VARUtil implements TimeSeriesPredictor {

    private final int p;
    private ArrayList<ArrayList<Double>> coeffs;
    
    // data normalization parameters (per feature dimension)
    private double[] dataMean;  // mean per feature
    private double[] dataStd;   // standard deviation per feature

    public VARUtil(int p) {
        this.p = p;
        this.coeffs = new ArrayList<>();
        this.dataMean = null;
        this.dataStd = null;
    }

    // Train the model with data to get coefficients
    public void fit(ArrayList<ArrayList<Double>> data) {
        int n = data.size();
        int k = data.get(0).size();

        // compute normalization parameters (per feature dimension)
        dataMean = new double[k];
        dataStd = new double[k];
        
        // compute mean and standard deviation per feature
        for (int col = 0; col < k; col++) {
            double sum = 0.0;
            for (int row = 0; row < n; row++) {
                sum += data.get(row).get(col);
            }
            dataMean[col] = sum / n;
            
            double sumSqDiff = 0.0;
            for (int row = 0; row < n; row++) {
                double diff = data.get(row).get(col) - dataMean[col];
                sumSqDiff += diff * diff;
            }
            dataStd[col] = Math.sqrt(sumSqDiff / n);
            
            // avoid division by zero; if std is 0 set to 1
            if (dataStd[col] < 1e-8) {
                dataStd[col] = 1.0;
            }
        }
        
        // normalize data
        ArrayList<ArrayList<Double>> normalizedData = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            ArrayList<Double> normalizedRow = new ArrayList<>();
            for (int j = 0; j < k; j++) {
                double normalizedValue = (data.get(i).get(j) - dataMean[j]) / dataStd[j];
                normalizedRow.add(normalizedValue);
            }
            normalizedData.add(normalizedRow);
        }

        ArrayList<ArrayList<Double>> X = new ArrayList<>();

        // construct the data matrix X (using normalized data)
        for (int i = p; i < n; i++) {
            ArrayList<Double> x = new ArrayList<>();
            for (int j = 0; j < p; j++) {
                x.addAll(normalizedData.get(i - j - 1));
            }
            X.add(x);
        }

        // compute the coefficients using OLS (using normalized data)
        Matrix Xmat = new Matrix(X);
        ArrayList<ArrayList<Double>> Yarry = new ArrayList<>();
        for (int i = p; i < n; i++) {
            Yarry.add(normalizedData.get(i));
        }
        Matrix Ymat = new Matrix(Yarry);
        Matrix XtX = Xmat.transpose().multiply(Xmat);
        Matrix XtY = Xmat.transpose().multiply(Ymat);
        Matrix beta = XtX.solve(XtY);
        this.coeffs = beta.transpose().getData();
    }

    // One step of prediction based on window. Window has p tuples.
    // Return the prediction result.
    public ArrayList<Double> predict(double[][] window) {
        if (dataMean == null || dataStd == null) {
            throw new IllegalStateException("Model not trained. Please call fit() first.");
        }
        
        int k = window[0].length;
        
        // normalize input window
        double[][] normalizedWindow = new double[p][k];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < k; j++) {
                normalizedWindow[i][j] = (window[i][j] - dataMean[j]) / dataStd[j];
            }
        }
        
        // build input vector (using normalized window)
        ArrayList<Double> x = new ArrayList<>();
        for (int i = 0; i < p; i++) {
            ArrayList<Double> tuple = new ArrayList<>();
            for (double value : normalizedWindow[p - i - 1]) {
                tuple.add(value);
            }
            x.addAll(tuple);
        }
        
        // predict (in normalized space)
        double[] yhat = new double[k];
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < x.size(); i++) {
                yhat[j] += x.get(i) * coeffs.get(j).get(i);
            }
        }
        
        // denormalize: convert prediction back to original scale
        ArrayList<Double> prediction_tuple = new ArrayList<>();
        for (int j = 0; j < k; j++) {
            double denormalizedValue = yhat[j] * dataStd[j] + dataMean[j];
            prediction_tuple.add(denormalizedValue);
        }

        return prediction_tuple;
    }

    @Override
    public int getWindowSize() {
        return p;
    }

    // Helper class for matrix operations
    public class Matrix {
        public final int m;
        public final int n;
        public final ArrayList<ArrayList<Double>> data;

        public Matrix(int m, int n) {
            this.m = m;
            this.n = n;
            this.data = new ArrayList<ArrayList<Double>>(m);
            for (int i = 0; i < m; i++) {
                ArrayList<Double> row = new ArrayList<Double>(n);
                for (int j = 0; j < n; j++) {
                    row.add(0.0);
                }
                this.data.add(row);
            }
        }

        public Matrix(ArrayList<ArrayList<Double>> data) {
            this.m = data.size();
            this.n = data.get(0).size();
            this.data = new ArrayList<ArrayList<Double>>(m);
            for (int i = 0; i < m; i++) {
                ArrayList<Double> row = new ArrayList<Double>(n);
                for (int j = 0; j < n; j++) {
                    row.add(data.get(i).get(j));
                }
                this.data.add(row);
            }
        }

        public Matrix transpose() {
            Matrix A = this;
            Matrix C = new Matrix(n, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    C.data.get(j).set(i, A.data.get(i).get(j));
                }
            }
            return C;
        }

        public Matrix multiply(Matrix B) {
            Matrix A = this;
            if (A.n != B.m) {
                throw new IllegalArgumentException("Matrix dimensions don't match");
            }
            Matrix C = new Matrix(A.m, B.n);
            for (int i = 0; i < C.m; i++) {
                for (int j = 0; j < C.n; j++) {
                    for (int k = 0; k < A.n; k++) {
                        C.data
                                .get(i)
                                .set(j, C.data.get(i).get(j) + A.data.get(i).get(k) * B.data.get(k).get(j));
                    }
                }
            }
            return C;
        }

        public ArrayList<ArrayList<Double>> getArray() {
            return data;
        }

        public Matrix solve(Matrix B) {
            Matrix A = this;
            if (A.m != A.n || A.m != B.m) {
                throw new IllegalArgumentException("Matrix dimensions don't match");
            }

            int n = A.n;
            Matrix[] LU = A.lu();
            Matrix L = LU[0];
            Matrix U = LU[1];

            // Solve LY = B using forward substitution
            Matrix Y = new Matrix(n, B.n);
            for (int j = 0; j < B.n; j++) {
                for (int i = 0; i < n; i++) {
                    Y.data.get(i).set(j, B.data.get(i).get(j));
                    for (int k = 0; k < i; k++) {
                        Y.data
                                .get(i)
                                .set(j, Y.data.get(i).get(j) - L.data.get(i).get(k) * Y.data.get(k).get(j));
                    }
                }
            }

            // Solve UX = Y using backward substitution
            Matrix X = new Matrix(n, B.n);
            for (int j = 0; j < B.n; j++) {
                for (int i = n - 1; i >= 0; i--) {
                    X.data.get(i).set(j, Y.data.get(i).get(j));
                    for (int k = i + 1; k < n; k++) {
                        X.data
                                .get(i)
                                .set(j, X.data.get(i).get(j) - U.data.get(i).get(k) * X.data.get(k).get(j));
                    }
                    X.data.get(i).set(j, X.data.get(i).get(j) / U.data.get(i).get(i));
                }
            }
            return X;
        }

        public Matrix[] lu() {
            Matrix A = this;
            if (A.m != A.n) {
                throw new IllegalArgumentException("Matrix dimensions don't match");
            }

            Matrix L = new Matrix(A.m, A.n);
            Matrix U = new Matrix(A.m, A.n);
            for (int j = 0; j < A.n; j++) {
                L.data.get(j).set(j, 1.0);
                for (int i = 0; i < j + 1; i++) {
                    double s1 = 0.0;
                    for (int k = 0; k < i; k++) {
                        s1 += U.data.get(k).get(j) * L.data.get(i).get(k);
                    }
                    U.data.get(i).set(j, A.data.get(i).get(j) - s1);
                }
                for (int i = j + 1; i < A.n; i++) {
                    double s2 = 0.0;
                    for (int k = 0; k < j; k++) {
                        s2 += U.data.get(k).get(j) * L.data.get(i).get(k);
                    }
                    L.data.get(i).set(j, (A.data.get(i).get(j) - s2) / U.data.get(j).get(j));
                }
            }
            return new Matrix[] {L, U};
        }

        public ArrayList<ArrayList<Double>> getData() {
            return data;
        }
    }
}
