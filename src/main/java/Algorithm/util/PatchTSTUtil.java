package Algorithm.util;

import java.util.ArrayList;
import java.io.*;

/**
 * PatchTST model implementation via Python script.
 * Uses PatchTST Transformer for time series prediction.
 */
public class PatchTSTUtil implements TimeSeriesPredictor {
    
    private final int p;  // window size (seq_len)
    private final int columnCnt;  // number of variables
    private final String pythonScriptPath;  // path to Python script
    private final String modelDir;  // directory to save/load models
    private final int d_model;  // Transformer dimension
    private final int n_heads;  // number of attention heads
    private final int e_layers;  // number of encoder layers
    private final int epochs;  // training epochs
    private final double learningRate;  // learning rate
    
    // prediction call counter for controlling output frequency
    private static int predictCallCount = 0;
    private static final int PREDICT_OUTPUT_INTERVAL = 100;  // output progress every N predictions
    
    public PatchTSTUtil(int p, int columnCnt) {
        this(p, columnCnt, "./python/patchtst_model.py", "./models", 128, 8, 3, 10, 0.001);
    }
    
    public PatchTSTUtil(int p, int columnCnt, String pythonScriptPath, String modelDir) {
        this(p, columnCnt, pythonScriptPath, modelDir, 128, 8, 3, 10, 0.001);
    }
    
    public PatchTSTUtil(int p, int columnCnt, String pythonScriptPath, String modelDir, 
                      int d_model, int n_heads, int e_layers, int epochs, double learningRate) {
        this.p = p;
        this.columnCnt = columnCnt;
        this.pythonScriptPath = pythonScriptPath;
        this.modelDir = modelDir;
        this.d_model = d_model;
        this.n_heads = n_heads;
        this.e_layers = e_layers;
        this.epochs = epochs;
        this.learningRate = learningRate;
        
        // ensure model directory exists
        File modelDirFile = new File(modelDir);
        if (!modelDirFile.exists()) {
            modelDirFile.mkdirs();
        }
    }
    
    @Override
    public void fit(ArrayList<ArrayList<Double>> data) {
        // reset prediction counter after new training
        synchronized (PatchTSTUtil.class) {
            predictCallCount = 0;
        }
        
        // delete old model file if present so new normalization params are used
        File oldModelFile = new File(modelDir, "patchtst_model.pkl");
        if (oldModelFile.exists()) {
            System.out.println("[PatchTST] Deleting old model file to ensure fresh training with normalization...");
            oldModelFile.delete();
        }
        
        System.out.println("[PatchTST] Starting model training...");
        System.out.println("[PatchTST] Training data size: " + data.size() + " samples, " + 
                          (data.isEmpty() ? 0 : data.get(0).size()) + " features");
        System.out.println("[PatchTST] Parameters: seq_len=" + p + ", epochs=" + epochs + 
                          ", learning_rate=" + learningRate + ", d_model=" + d_model + 
                          ", n_heads=" + n_heads + ", e_layers=" + e_layers);
        
        File tempInputFile = null;
        File tempOutputFile = null;
        try {
            // create temp files for input/output
            System.out.println("[PatchTST] Creating temporary files...");
            tempInputFile = File.createTempFile("patchtst_input_", ".json");
            tempOutputFile = File.createTempFile("patchtst_output_", ".json");
            tempInputFile.deleteOnExit();
            tempOutputFile.deleteOnExit();
            
            // build JSON input and write to temp file
            System.out.println("[PatchTST] Writing training data to temporary file...");
            try (BufferedWriter writer = new BufferedWriter(
                    new FileWriter(tempInputFile))) {
                writer.write("{\n");
                writer.write("  \"data\": [\n");
                for (int i = 0; i < data.size(); i++) {
                    writer.write("    [");
                    for (int j = 0; j < data.get(i).size(); j++) {
                        writer.write(String.valueOf(data.get(i).get(j)));
                        if (j < data.get(i).size() - 1) {
                            writer.write(", ");
                        }
                    }
                    writer.write("]");
                    if (i < data.size() - 1) {
                        writer.write(",");
                    }
                    writer.write("\n");
                }
                writer.write("  ],\n");
                writer.write("  \"seq_len\": ");
                writer.write(String.valueOf(p));
                writer.write(",\n");
                writer.write("  \"pred_len\": 1,\n");
                writer.write("  \"d_model\": ");
                writer.write(String.valueOf(d_model));
                writer.write(",\n");
                writer.write("  \"n_heads\": ");
                writer.write(String.valueOf(n_heads));
                writer.write(",\n");
                writer.write("  \"e_layers\": ");
                writer.write(String.valueOf(e_layers));
                writer.write(",\n");
                writer.write("  \"epochs\": ");
                writer.write(String.valueOf(epochs));
                writer.write(",\n");
                writer.write("  \"learning_rate\": ");
                writer.write(String.valueOf(learningRate));
                writer.write(",\n");
                writer.write("  \"model_dir\": \"");
                writer.write(modelDir);
                writer.write("\"\n");
                writer.write("}\n");
            }
            System.out.println("[PatchTST] Training data file created: " + tempInputFile.getAbsolutePath());
            
            // call Python script with temp file
            // force arm64 when needed (work around architecture mismatch)
            System.out.println("[PatchTST] Calling Python script for training...");
            String osArch = System.getProperty("os.arch");
            ProcessBuilder pb;
            if (osArch.equals("aarch64") || osArch.equals("arm64")) {
                // system is arm64, use arch command to force arm64
                pb = new ProcessBuilder("arch", "-arm64", "python3", pythonScriptPath, "fit", 
                        tempInputFile.getAbsolutePath());
            } else {
                // other architectures, run directly
                pb = new ProcessBuilder("python3", pythonScriptPath, "fit", 
                        tempInputFile.getAbsolutePath());
            }
            // set env for arm64 when needed
            pb.environment().put("ARCH", "arm64");
            pb.environment().put("ARCHFLAGS", "-arch arm64");
            // do not merge error stream; read separately for easier diagnosis
            pb.redirectErrorStream(false);
            Process process = pb.start();
            System.out.println("[PatchTST] Python process started, waiting for training to complete...");
            
            // use threads to read output (avoid blocking)
            StringBuilder output = new StringBuilder();
            StringBuilder errorOutput = new StringBuilder();
            
            Thread outputThread = new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        output.append(line).append("\n");
                    }
                } catch (IOException e) {
                    errorOutput.append("Failed to read output: ").append(e.getMessage()).append("\n");
                }
            });
            
            Thread errorThread = new Thread(() -> {
                try (BufferedReader errorReader = new BufferedReader(
                        new InputStreamReader(process.getErrorStream()))) {
                    String line;
                    while ((line = errorReader.readLine()) != null) {
                        errorOutput.append(line).append("\n");
                        // print stderr in real time (training progress)
                        if (line.contains("Epoch") || line.contains("Starting") || 
                            line.contains("Creating") || line.contains("Building") || 
                            line.contains("Processed") || line.contains("Sampling") ||
                            line.contains("Training data prepared") || line.contains("Starting training loop") ||
                            line.contains("Training completed") || line.contains("Saving model") ||
                            line.contains("Warning:") || line.contains("Device:")) {
                            System.out.println("[PatchTST Python] " + line);
                        }
                    }
                } catch (IOException e) {
                    // ignore
                }
            });
            
            outputThread.start();
            errorThread.start();
            
            int exitCode = process.waitFor();
            
            // wait for reader threads to finish
            System.out.println("[PatchTST] Reading Python output...");
            try {
                outputThread.join(30000); // wait at most 30 seconds
                errorThread.join(30000);
            } catch (InterruptedException e) {
                // ignore
            }
            
            // show Python stdout (training progress)
            String pythonOutput = output.toString().trim();
            String pythonErrors = errorOutput.toString().trim();
            
            // show device acceleration info first
            if (!pythonErrors.isEmpty()) {
                String[] lines = pythonErrors.split("\n");
                for (String line : lines) {
                    if (line.contains("MPS (Metal) acceleration enabled") || 
                        line.contains("CUDA acceleration enabled") ||
                        line.contains("Using CPU") ||
                        line.contains("Device:")) {
                        System.out.println("[PatchTST] " + line);
                    }
                }
            }
            
            if (!pythonOutput.isEmpty()) {
                // print Python progress (from stderr, in errorOutput)
                if (!pythonErrors.isEmpty() && !pythonErrors.contains("error")) {
                    // show training progress (normal stderr output)
                    System.out.println("[PatchTST] Training Progress:");
                    String[] lines = pythonErrors.split("\n");
                    for (String line : lines) {
                        if (line.contains("Epoch") || line.contains("Starting") || line.contains("Training") || 
                            line.contains("completed") || line.contains("Final loss") ||
                            line.contains("Data statistics") || line.contains("Normalized data")) {
                            System.out.println("  " + line);
                        }
                    }
                }
            }
            
            if (exitCode != 0) {
                // print detailed error info
                System.err.println("==========================================");
                System.err.println("[PatchTST] Python script execution failed (exit code: " + exitCode + ")");
                System.err.println("==========================================");
                System.err.println("Standard output:");
                System.err.println(output.toString());
                System.err.println("Error output:");
                System.err.println(errorOutput.toString());
                System.err.println("==========================================");
                
                // try to extract JSON error from output
                String errorJson = output.toString().trim();
                if (errorJson.isEmpty()) {
                    errorJson = errorOutput.toString().trim();
                }
                
                // if output is JSON, try to parse error
                if (errorJson.contains("\"error\"")) {
                    // simple extraction of error message
                    int errorStart = errorJson.indexOf("\"error\"");
                    if (errorStart != -1) {
                        int colonIdx = errorJson.indexOf(":", errorStart);
                        int quoteStart = errorJson.indexOf("\"", colonIdx);
                        int quoteEnd = errorJson.indexOf("\"", quoteStart + 1);
                        if (quoteEnd > quoteStart) {
                            String errorMsg = errorJson.substring(quoteStart + 1, quoteEnd);
                            throw new RuntimeException("[PatchTST] Python script error: " + errorMsg + "\nFull output:\n" + output.toString() + "\nError output:\n" + errorOutput.toString());
                        }
                    }
                }
                
                throw new RuntimeException("[PatchTST] Python script execution failed (exit code: " + exitCode + ")\nStandard output:\n" + output.toString() + "\nError output:\n" + errorOutput.toString());
            }
            
            // read result from output file (if file output is used)
            String result = output.toString().trim();
            if (tempOutputFile.exists() && tempOutputFile.length() > 0) {
                try (BufferedReader reader = new BufferedReader(
                        new FileReader(tempOutputFile))) {
                    result = reader.readLine();
                }
            }
            
            // check if result indicates success
            if (result.contains("\"status\":\"success\"") || result.contains("\"status\": \"success\"")) {
                // success
                System.out.println("[PatchTST] Model training completed successfully!");
            } else if (!result.isEmpty()) {
                // result not empty but no success; may be warning
                System.err.println("[PatchTST] Warning: Python script output: " + result);
            }
            
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("[PatchTST] Failed to call Python script: " + e.getMessage(), e);
        } finally {
            // clean up temp files
            if (tempInputFile != null && tempInputFile.exists()) {
                tempInputFile.delete();
            }
            if (tempOutputFile != null && tempOutputFile.exists()) {
                tempOutputFile.delete();
            }
        }
    }
    
    @Override
    public ArrayList<Double> predict(double[][] window) {
        // increment prediction call count
        synchronized (PatchTSTUtil.class) {
            predictCallCount++;
        }
        
        // control output frequency: first call and every N-th call
        boolean shouldOutput = (predictCallCount == 1) || (predictCallCount % PREDICT_OUTPUT_INTERVAL == 0);
        
        if (shouldOutput && predictCallCount == 1) {
            System.out.println("[PatchTST] Starting prediction phase...");
        } else if (shouldOutput) {
            System.out.println("[PatchTST] Prediction progress: " + predictCallCount + " predictions completed");
        }
        
        if (window.length != p) {
            throw new IllegalArgumentException("Window size must be " + p + ", actual: " + window.length);
        }
        if (window[0].length != columnCnt) {
            throw new IllegalArgumentException("Variable count must be " + columnCnt + ", actual: " + window[0].length);
        }
        
        File tempInputFile = null;
        try {
            // create temp file for input (predict payload is small; use file for consistency)
            tempInputFile = File.createTempFile("patchtst_predict_input_", ".json");
            tempInputFile.deleteOnExit();
            
            // build JSON input and write to temp file
            try (BufferedWriter writer = new BufferedWriter(
                    new FileWriter(tempInputFile))) {
                writer.write("{\n");
                writer.write("  \"window\": [\n");
                for (int i = 0; i < window.length; i++) {
                    writer.write("    [");
                    for (int j = 0; j < window[i].length; j++) {
                        writer.write(String.valueOf(window[i][j]));
                        if (j < window[i].length - 1) {
                            writer.write(", ");
                        }
                    }
                    writer.write("]");
                    if (i < window.length - 1) {
                        writer.write(",");
                    }
                    writer.write("\n");
                }
                writer.write("  ],\n");
                writer.write("  \"seq_len\": ");
                writer.write(String.valueOf(p));
                writer.write(",\n");
                writer.write("  \"pred_len\": 1,\n");
                writer.write("  \"model_dir\": \"");
                writer.write(modelDir);
                writer.write("\"\n");
                writer.write("}\n");
            }
            
            // call Python script with temp file
            // force arm64 when needed (work around architecture mismatch)
            String osArch = System.getProperty("os.arch");
            ProcessBuilder pb;
            if (osArch.equals("aarch64") || osArch.equals("arm64")) {
                // system is arm64, use arch command to force arm64
                pb = new ProcessBuilder("arch", "-arm64", "python3", pythonScriptPath, "predict", 
                        tempInputFile.getAbsolutePath());
            } else {
                // other architectures, run directly
                pb = new ProcessBuilder("python3", pythonScriptPath, "predict", 
                        tempInputFile.getAbsolutePath());
            }
            // set env for arm64 when needed
            pb.environment().put("ARCH", "arm64");
            pb.environment().put("ARCHFLAGS", "-arch arm64");
            // do not merge error stream; read separately for easier diagnosis
            pb.redirectErrorStream(false);
            Process process = pb.start();
            
            // use threads to read output (avoid blocking)
            StringBuilder output = new StringBuilder();
            StringBuilder errorOutput = new StringBuilder();
            
            Thread outputThread = new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        output.append(line).append("\n");
                    }
                } catch (IOException e) {
                    errorOutput.append("Failed to read output: ").append(e.getMessage()).append("\n");
                }
            });
            
            Thread errorThread = new Thread(() -> {
                try (BufferedReader errorReader = new BufferedReader(
                        new InputStreamReader(process.getErrorStream()))) {
                    String line;
                    while ((line = errorReader.readLine()) != null) {
                        errorOutput.append(line).append("\n");
                    }
                } catch (IOException e) {
                    // ignore
                }
            });
            
            outputThread.start();
            errorThread.start();
            
            int exitCode = process.waitFor();
            
            // wait for reader threads
            try {
                outputThread.join(10000); // wait at most 10 seconds
                errorThread.join(10000);
            } catch (InterruptedException e) {
                // ignore
            }
            
            // show device acceleration info (only on first prediction)
            if (predictCallCount == 1) {
                String pythonErrors = errorOutput.toString().trim();
                if (!pythonErrors.isEmpty()) {
                    String[] lines = pythonErrors.split("\n");
                    for (String line : lines) {
                        if (line.contains("MPS (Metal) acceleration enabled") || 
                            line.contains("CUDA acceleration enabled") ||
                            line.contains("Using CPU") ||
                            line.contains("Prediction device:")) {
                            System.out.println("[PatchTST] " + line);
                        }
                    }
                }
            }
            
            if (exitCode != 0) {
                // print detailed error info
                System.err.println("==========================================");
                System.err.println("[PatchTST] Python script execution failed (exit code: " + exitCode + ")");
                System.err.println("==========================================");
                System.err.println("Standard output:");
                System.err.println(output.toString());
                System.err.println("Error output:");
                System.err.println(errorOutput.toString());
                System.err.println("==========================================");
                
                // try to extract JSON error from output
                String errorJson = output.toString().trim();
                if (errorJson.isEmpty()) {
                    errorJson = errorOutput.toString().trim();
                }
                
                // if output is JSON, try to parse error
                if (errorJson.contains("\"error\"")) {
                    // simple extraction of error message
                    int errorStart = errorJson.indexOf("\"error\"");
                    if (errorStart != -1) {
                        int colonIdx = errorJson.indexOf(":", errorStart);
                        int quoteStart = errorJson.indexOf("\"", colonIdx);
                        int quoteEnd = errorJson.indexOf("\"", quoteStart + 1);
                        if (quoteEnd > quoteStart) {
                            String errorMsg = errorJson.substring(quoteStart + 1, quoteEnd);
                            throw new RuntimeException("[PatchTST] Python script error: " + errorMsg + "\nFull output:\n" + output.toString() + "\nError output:\n" + errorOutput.toString());
                        }
                    }
                }
                
                throw new RuntimeException("[PatchTST] Python script execution failed (exit code: " + exitCode + ")\nStandard output:\n" + output.toString() + "\nError output:\n" + errorOutput.toString());
            }
            
            // parse JSON output
            String result = output.toString().trim();
            
            // simple JSON parsing (extract prediction array); for complex JSON use Jackson or Gson
            ArrayList<Double> prediction = parsePredictionFromJson(result);
            
            // print completion on last prediction of batch
            if (shouldOutput && predictCallCount % PREDICT_OUTPUT_INTERVAL == 0) {
                System.out.println("[PatchTST] Prediction batch completed successfully");
            }
            
            return prediction;
            
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("[PatchTST] Failed to call Python script: " + e.getMessage(), e);
        } finally {
            // clean up temp files
            if (tempInputFile != null && tempInputFile.exists()) {
                tempInputFile.delete();
            }
        }
    }
    
    /**
     * Parse prediction array from JSON string.
     * Simple implementation assuming format {"prediction": [1.0, 2.0, ...]}.
     * For complex JSON consider using Jackson or Gson.
     */
    private ArrayList<Double> parsePredictionFromJson(String json) {
        ArrayList<Double> result = new ArrayList<>();
        
        // remove whitespace for parsing
        String cleanJson = json.replaceAll("\\s+", "");
        
        // find start and end of prediction array
        int startIdx = cleanJson.indexOf("\"prediction\":[");
        if (startIdx == -1) {
            // try alternative format
            startIdx = cleanJson.indexOf("prediction:[");
            if (startIdx == -1) {
                throw new RuntimeException("[PatchTST] Cannot parse Python output, prediction field not found: " + json);
            }
            startIdx = cleanJson.indexOf("[", startIdx) + 1;
        } else {
            startIdx = cleanJson.indexOf("[", startIdx) + 1;
        }
        
        // find matching closing bracket (handle nesting)
        int endIdx = startIdx;
        int bracketCount = 1;
        while (endIdx < cleanJson.length() && bracketCount > 0) {
            if (cleanJson.charAt(endIdx) == '[') {
                bracketCount++;
            } else if (cleanJson.charAt(endIdx) == ']') {
                bracketCount--;
            }
            endIdx++;
        }
        endIdx--; // back to last ']'
        
        if (bracketCount != 0) {
            throw new RuntimeException("[PatchTST] Cannot parse Python output, bracket mismatch: " + json);
        }
        
        // handle empty array
        if (startIdx >= endIdx) {
            return result;
        }
        
        // split and parse numbers
        String arrayContent = cleanJson.substring(startIdx, endIdx);
        String[] values = arrayContent.split(",");
        for (String value : values) {
            value = value.trim();
            if (value.isEmpty()) {
                continue;
            }
            try {
                result.add(Double.parseDouble(value));
            } catch (NumberFormatException e) {
                throw new RuntimeException("[PatchTST] Cannot parse number: " + value + " (from: " + json + ")", e);
            }
        }
        
        return result;
    }
    
    @Override
    public int getWindowSize() {
        return p;
    }
    
    /**
     * Get Python script path.
     */
    public String getPythonScriptPath() {
        return pythonScriptPath;
    }
    
    /**
     * Get model directory.
     */
    public String getModelDir() {
        return modelDir;
    }
    
    /**
     * Check if Python environment and dependencies are available.
     */
    public static boolean checkPythonEnvironment(String pythonScriptPath) {
        try {
            File scriptFile = new File(pythonScriptPath);
            if (!scriptFile.exists()) {
                System.err.println("[PatchTST] Python script not found: " + pythonScriptPath);
                return false;
            }
            
            // check if Python is available
            System.out.println("[PatchTST] Checking Python environment...");
            ProcessBuilder pb = new ProcessBuilder("python3", "--version");
            pb.redirectErrorStream(true);
            Process process = pb.start();
            
            // read output to avoid blocking process
            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                }
            }
            
            int exitCode = process.waitFor();
            
            if (exitCode != 0) {
                System.err.println("[PatchTST] Python3 is not available");
                return false;
            }
            System.out.println("[PatchTST] Python version: " + output.toString().trim());
            
            // check if PyTorch is installed
            // force arm64 for Python when needed
            String osArch = System.getProperty("os.arch");
            ProcessBuilder torchPb;
            if (osArch.equals("aarch64") || osArch.equals("arm64")) {
                torchPb = new ProcessBuilder("arch", "-arm64", "python3", "-c", 
                        "import torch; print('PyTorch version:', torch.__version__)");
            } else {
                torchPb = new ProcessBuilder("python3", "-c", 
                        "import torch; print('PyTorch version:', torch.__version__)");
            }
            // set env for arm64
            torchPb.environment().put("ARCH", "arm64");
            torchPb.environment().put("ARCHFLAGS", "-arch arm64");
            // do not redirectErrorStream; read stdout and stderr separately
            torchPb.redirectErrorStream(false);
            Process torchProcess = torchPb.start();
            
            // read output to avoid blocking
            StringBuilder torchOutput = new StringBuilder();
            StringBuilder torchError = new StringBuilder();
            
            // use threads to read stdout and stderr (avoid blocking)
            Thread torchOutputThread = new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(torchProcess.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        torchOutput.append(line).append("\n");
                    }
                } catch (IOException e) {
                    // ignore
                }
            });
            
            Thread torchErrorThread = new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(torchProcess.getErrorStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        torchError.append(line).append("\n");
                    }
                } catch (IOException e) {
                    // ignore
                }
            });
            
            torchOutputThread.start();
            torchErrorThread.start();
            
            int torchExitCode = torchProcess.waitFor();
            
            // wait for reader threads
            try {
                torchOutputThread.join(5000); // wait at most 5 seconds
                torchErrorThread.join(5000);
            } catch (InterruptedException e) {
                // ignore
            }
            
            String version = torchOutput.toString().trim();
            String errorMsg = torchError.toString().trim();
            
            if (torchExitCode != 0) {
                // check for architecture mismatch
                if (errorMsg.contains("incompatible architecture") || errorMsg.contains("mach-o") || 
                    errorMsg.contains("arm64") || errorMsg.contains("x86_64")) {
                    System.err.println("[PatchTST] Warning: PyTorch architecture mismatch!");
                    System.err.println("PyTorch is installed but architecture is incompatible (arm64/x86_64 mismatch)");
                    System.err.println("");
                    System.err.println("Solutions:");
                    System.err.println("1. If Java runs under Rosetta (x86_64), install x86_64 PyTorch:");
                    System.err.println("   arch -x86_64 pip3 install torch");
                    System.err.println("");
                    System.err.println("2. Or ensure both Java and Python use arm64 architecture (recommended):");
                    System.err.println("   Reinstall arm64 PyTorch: pip3 install torch");
                    System.err.println("");
                    System.err.println("3. Check current Python architecture:");
                    System.err.println("   python3 -c \"import platform; print(platform.machine())\"");
                    System.err.println("");
                    System.err.println("Note: Program will continue, but PatchTST model may not work properly.");
                } else {
                    System.err.println("[PatchTST] Warning: PyTorch not installed or import failed, run: pip install -r python/requirements.txt");
                    if (!errorMsg.isEmpty() && !errorMsg.contains("incompatible architecture")) {
                        System.err.println("[PatchTST] Error message: " + errorMsg);
                    }
                }
            } else {
                // PyTorch installed; show version
                if (!version.isEmpty()) {
                    System.out.println("[PatchTST] PyTorch installed: " + version);
                } else if (!errorMsg.isEmpty()) {
                    // stdout empty but stderr has content; may be warning
                    System.out.println("[PatchTST] PyTorch check completed (may have warnings)");
                }
            }
            
            // return true even if PyTorch check failed, so program can continue
            // architecture mismatch may be only a warning; runtime may still work
            return true;
        } catch (Exception e) {
            System.err.println("[PatchTST] Failed to check Python environment: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
}
