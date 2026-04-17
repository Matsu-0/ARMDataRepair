import Algorithm.util.KDTreeUtil;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

public class LoadData {
    // return values
    private final ArrayList<ArrayList<Double>> td_clean = new ArrayList<>();
    private final ArrayList<ArrayList<Double>> td_raw = new ArrayList<>();
    private final ArrayList<ArrayList<Double>> md = new ArrayList<>();
    private final ArrayList<ArrayList<Double>> md_complete = new ArrayList<>();
    private final ArrayList<Long> td_time = new ArrayList<>();

    private final ArrayList<String> td_time_str = new ArrayList<>();

    // parameter
    private final int td_len;
    private final int md_len;
    private final double eta;
    private final int md_len_complete;
    private final Random random;

    // array
    private double[][] td_clean_array;
    private double[][] td_raw_array;
    private double[][] md_array;
    private double[][] md_array_complete;
    private long[] td_time_array;

    // standard
    private double[] mean;
    private double[] std;

    // kdTree
    private KDTreeUtil kdTree;
    private KDTreeUtil kdTreeComplete;

    public LoadData(String td_path, String md_path, int td_len, int md_len, double eta, int seed) throws Exception {
        this.td_len = td_len;
        this.md_len = md_len;
        this.eta = eta;
        this.md_len_complete = Integer.parseInt(md_path.split("_")[2].split("\\.")[0]);
        this.random = new Random(seed);

        this.loadMasterDataComplete(md_path);
        this.loadMasterData(md_path);
        kdTreeComplete = new KDTreeUtil(md_array_complete);
        kdTree = new KDTreeUtil(md_array);
        this.loadTimeSeriesData(td_path);
    }

    private void loadMasterDataComplete(String filename) throws Exception {  // full random sample master
        Scanner sc = new Scanner(new File(filename));
        sc.useDelimiter("\\s*([,\\r\\n])\\s*"); // set separator
        sc.nextLine();

        // add value
        while (sc.hasNextLine()) {
            String[] line_str = (sc.nextLine()).split(",");
            addValues(this.md_complete, line_str);
        }
        fillNullValue(this.md_complete);  // should not contain null

        this.md_array_complete = getDoubleArray(md_complete);
        calMeanStd(md_array_complete);
//        standardization(md_array_complete);
    }

    private void loadMasterData(String filename) throws Exception {  // full random sample master
        Scanner sc = new Scanner(new File(filename));
        sc.useDelimiter("\\s*([,\\r\\n])\\s*"); // set separator
        sc.nextLine();

        // sample num + rev
        boolean rev = false;
        int sample_num = Math.min(md_len, md_len_complete);
        if (sample_num > md_len_complete / 2) {
            rev = true;
            sample_num = md_len_complete - sample_num;
        }

        // sampling
        HashSet<Integer> hs = new HashSet<>();
        int[] idx = new int[md_len_complete];
        for (int j = 0; j < md_len_complete; idx[j] = j, j++) ;
        for (int i = 0, tmp, randomIndex; i < sample_num; i++) {
            randomIndex = md_len_complete - 1 - random.nextInt(md_len_complete - i);
            tmp = idx[randomIndex];
            hs.add(tmp);
            idx[randomIndex] = idx[i];
            idx[i] = tmp;
        }

        // add value
        for (int k = 0; sc.hasNextLine(); k++) {
            String[] line_str = (sc.nextLine()).split(",");
            if (rev == hs.contains(k))  // sampling
                continue;
            addValues(this.md, line_str);
        }
        fillNullValue(this.md);  // should not contain null

        this.md_array = getDoubleArray(md);
//        standardization(md_array);
    }

    private void loadTimeSeriesData(String filename) throws Exception {
        Scanner sc = new Scanner(new File(filename));
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        sc.useDelimiter("\\s*([,\\r\\n])\\s*"); // set separator
        sc.nextLine();  // skip table header
        for (int k = td_len; k > 0 && sc.hasNextLine(); --k) {  // the size of td_clean is dataLen
            String new_line = sc.nextLine();

            if (new_line.charAt(new_line.length()-1) == ',') {
                new_line = new_line + '0';
            }
            String[] line_str = (new_line).split(",");
            // td_time
            this.td_time.add(format.parse(line_str[0]).getTime());
            this.td_time_str.add(line_str[0]);
//            System.out.println(Arrays.toString(line_str));
            // td_clean
            addValues(this.td_clean, line_str);
            // td_raw
            addValues(this.td_raw, line_str);
        }
//        System.out.println(this.td_clean.get(0));
        fillNullValue(this.td_clean);
        fillNullValue(this.td_raw);

        this.td_clean_array = getDoubleArray(td_clean);
        this.td_raw_array = getDoubleArray(td_raw);
        this.td_time_array = getLongArray(td_time);

//        standardization(td_clean_array);
//        standardization(td_raw_array);

//        double[] td_row;
//        double dist;
//        for (double[] td_clean_tuple : td_clean_array) {
//            dist = kdTreeComplete.nearestNeighborDistance(td_clean_tuple);
//            if (dist > eta) {
//                td_row = kdTreeComplete.nearestNeighbor(td_clean_tuple);
//                System.arraycopy(td_row, 0, td_clean_tuple, 0, td_clean_tuple.length);
//            }
//        }
    }

    public String[] getTd_time_str() {
        return getStringArray(td_time_str);
    }

    public double[][] getTd_raw_array() {
        return td_raw_array;
    }

    public long[] getTd_time() {
        return this.td_time_array;
    }

    public double[][] getTd_clean() {
        return this.td_clean_array;
    }

    public double[][] getMd() {
        return this.md_array;
    }

    public KDTreeUtil getKdTree() {
        return this.kdTree;
    }

    public KDTreeUtil getKdTreeComplete() {
        return this.kdTreeComplete;
    }

    private void addValues(ArrayList<ArrayList<Double>> array, String[] line) {
        ArrayList<Double> values = new ArrayList<>();
        String value;
        for (int i = 1; i < line.length; ++i) {
            value = line[i];
            if (!value.equals("")) {
                values.add(Double.parseDouble(value));
            } else {
                values.add(Double.NaN);
            }
        }
        array.add(values);
    }

    private void standardization(double[][] values) {
        int row_len = values.length;
        int col_len = values[0].length;

        double[] temp = new double[row_len];
        double[] standardizedArr;
        for (int col = 0; col < col_len; col++) {
            for (int row = 0; row < row_len; row++) {
                temp[row] = values[row][col];
            }
            int finalCol = col;
            standardizedArr = Arrays.stream(temp).map(x -> (x - mean[finalCol]) / std[finalCol]).toArray();
            for (int row = 0; row < row_len; row++) {
                values[row][col] = standardizedArr[row];
            }
        }
    }

    private void calMeanStd(double[][] values) {
        int row_len = values.length;
        int col_len = values[0].length;
        this.mean = new double[col_len];
        this.std = new double[col_len];

        double[] temp = new double[row_len];
        for (int col = 0; col < col_len; col++) {
            for (int row = 0; row < row_len; row++) {
                temp[row] = values[row][col];
            }
            mean[col] = Arrays.stream(temp).sum() / temp.length;
            int finalCol = col;
            std[col] = Math.sqrt(Arrays.stream(temp).map(x -> Math.pow(x - mean[finalCol], 2)).sum() / temp.length);
        }
    }

    private void fillNullValue(ArrayList<ArrayList<Double>> values) {
        double value;
        for (int col = 0; col < values.get(0).size(); col++) {
            value = values.get(0).get(col);
            for (ArrayList<Double> rowArray : values) {
//                System.out.println(rowArray.toString());
//                System.out.println(col);
                if (Double.isNaN(rowArray.get(col))) {
                    rowArray.set(col, value);
                } else {
                    value = rowArray.get(col);
                }
            }
        }
    }

    private long[] getLongArray(ArrayList<Long> arrayList) {
        long[] rtn = new long[arrayList.size()];
        for (int i = 0; i < arrayList.size(); ++i)
            rtn[i] = arrayList.get(i);
        return rtn;
    }


    private String[] getStringArray(ArrayList<String> arrayList) {
        String[] rtn = new String[arrayList.size()];
        for (int i = 0; i < arrayList.size(); ++i)
            rtn[i] = arrayList.get(i);
        return rtn;
    }

    private double[][] getDoubleArray(ArrayList<ArrayList<Double>> arrayList) {
        double[][] rtn = new double[arrayList.size()][arrayList.get(0).size()];
        for (int i = 0, j; i < arrayList.size(); ++i)
            for (j = 0; j < arrayList.get(0).size(); ++j)
                rtn[i][j] = arrayList.get(i).get(j);
        return rtn;
    }

    public static void writeToFile(double[][] td, String targetFileName) throws Exception {
        File writeFile = new File(targetFileName);
        BufferedWriter writeText = new BufferedWriter(new FileWriter(writeFile));

        writeText.write("timestamp");
        for (int i = 0; i < td[0].length; i++) {
            writeText.write(",value" + i);
        }

        for (int row = 0; row < td.length; row++) {
            writeText.newLine();
            writeText.write(Integer.toString(row));
            for (int col = 0; col < td[0].length; col++) {
                writeText.write("," + td[row][col]);
            }
        }
        writeText.flush();
        writeText.close();
    }
}
