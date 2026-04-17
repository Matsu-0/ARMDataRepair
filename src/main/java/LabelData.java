import java.util.Random;
import java.util.HashSet;

public class LabelData {
    private final HashSet<Integer> hs = new HashSet<>();

    private final double[][] td_clean;
    private final double[][] td_dirty;
    private final double[][] td_label;
    private final boolean[] td_bool;

    private final Random random;


    public LabelData(double[][] td_clean, double[][] td_dirty, double rate, int seed) {
        this.td_clean = td_clean;
        this.td_dirty = td_dirty;
        this.td_label = new double[td_clean.length][td_clean[0].length];
        this.td_bool = new boolean[td_clean.length];

        this.random = new Random(seed);

        selectM(td_clean.length, (int) (td_clean.length * rate));
        labelData();
    }

    private void selectM(int n, int m) {
        int[] idx = new int[n];
        for (int j = 0; j < n; idx[j] = j, j++) ;

        hs.add(0);
        hs.add(1);
        hs.add(2);
        for (int i = 0, tmp, randomIndex; i < m - 3; i++) {
            randomIndex = n - 1 - random.nextInt(n - i);
            tmp = idx[randomIndex];
            hs.add(tmp);
            idx[randomIndex] = idx[i];
            idx[i] = tmp;
        }
    }

    private void labelData() {
        for (int row = 0; row < td_clean.length; row++) {
            if (hs.contains(row)) {
                td_bool[row] = true;
                System.arraycopy(td_clean[row], 0, td_label[row], 0, td_clean[0].length);
            } else {
                td_bool[row] = false;
                System.arraycopy(td_dirty[row], 0, td_label[row], 0, td_dirty[0].length);
            }
        }
    }

    public double[][] getTd_label() {
        return td_label;
    }

    public boolean[] getTd_bool() {
        return td_bool;
    }

}
