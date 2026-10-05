package Algorithm;

import java.io.*;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Baseline: CSDI (Tashiro et al., NeurIPS 2021) self-detects errors via
 * reconstruction residual, then imputes the flagged points. Does not use ARM's
 * error set E, domain constraints, or clean labels.
 *
 * Official CSDI sources live under {@code baseline/CSDI}; the ARM adapter is
 * {@code python/csdi_repair.py}.
 */
public class CSDIRepair {
    private final double[][] td_repaired;
    private final long cost_time;

    public CSDIRepair(double[][] td, long[] td_time, int columnCnt) {
        this(td, td_time, columnCnt, "./python/csdi_repair.py", "./baseline/CSDI", "./models/csdi");
    }

    public CSDIRepair(double[][] td, long[] td_time, int columnCnt,
                      String pythonScriptPath, String csdiRoot, String workdir) {
        this.td_repaired = new double[td.length][columnCnt];
        File dir = new File(workdir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        long start = System.nanoTime();
        this.repair(td, pythonScriptPath, csdiRoot, workdir);
        this.cost_time = (System.nanoTime() - start) / 1_000_000L;
        System.out.println("CSDIRepair time cost:" + cost_time + "ms");
    }

    public double[][] getTd_repaired() {
        return td_repaired;
    }

    public long getCost_time() {
        return cost_time;
    }

    private void repair(double[][] td, String pythonScriptPath, String csdiRoot, String workdir) {
        File input = new File(workdir, "dirty.csv");
        File output = new File(workdir, "repaired.csv");
        writeMatrix(input, td);

        String osArch = System.getProperty("os.arch");
        ProcessBuilder pb;
        List<String> cmd = new ArrayList<>();
        if (osArch.equals("aarch64") || osArch.equals("arm64")) {
            cmd.add("arch");
            cmd.add("-arm64");
        }
        cmd.add("python3");
        cmd.add(pythonScriptPath);
        cmd.add("--input");
        cmd.add(input.getAbsolutePath());
        cmd.add("--output");
        cmd.add(output.getAbsolutePath());
        cmd.add("--csdi_root");
        cmd.add(csdiRoot);
        cmd.add("--workdir");
        cmd.add(workdir);
        pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        pb.environment().put("PYTHONUNBUFFERED", "1");

        System.out.println("[CSDI] " + String.join(" ", cmd));
        try {
            Process process = pb.start();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println("[CSDI] " + line);
                }
            }
            int code = process.waitFor();
            if (code != 0) {
                throw new RuntimeException("CSDI python exited with code " + code);
            }
            readMatrix(output, td_repaired);
        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new RuntimeException("CSDIRepair failed: " + e.getMessage(), e);
        }
    }

    private static void writeMatrix(File file, double[][] data) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))) {
            for (double[] row : data) {
                StringBuilder line = new StringBuilder();
                for (int c = 0; c < row.length; c++) {
                    if (c > 0) {
                        line.append(',');
                    }
                    line.append(String.format(Locale.US, "%.10f", row[c]));
                }
                bw.write(line.toString());
                bw.newLine();
            }
        } catch (IOException e) {
            throw new RuntimeException("failed to write " + file, e);
        }
    }

    private static void readMatrix(File file, double[][] dest) {
        try {
            List<String> lines = Files.readAllLines(file.toPath());
            if (lines.size() != dest.length) {
                throw new RuntimeException("CSDI output rows " + lines.size() + " != " + dest.length);
            }
            for (int i = 0; i < dest.length; i++) {
                String[] parts = lines.get(i).trim().split(",");
                if (parts.length != dest[i].length) {
                    throw new RuntimeException("CSDI output cols " + parts.length + " != " + dest[i].length
                            + " at row " + i);
                }
                for (int c = 0; c < dest[i].length; c++) {
                    dest[i][c] = Double.parseDouble(parts[c].trim());
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("failed to read " + file, e);
        }
    }
}
