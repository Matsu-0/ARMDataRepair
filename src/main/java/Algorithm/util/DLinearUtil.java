package Algorithm.util;

import java.util.ArrayList;
import java.io.*;

/**
 * DLinear model implementation via Python script.
 * Uses the DLinear deep learning model for time series prediction.
 */
public class DLinearUtil implements TimeSeriesPredictor {
    
    private final int p;  // window size (seq_len)
    private final int columnCnt;  // number of variables
    private final String pythonScriptPath;  // path to Python script
    private final String modelDir;  // directory to save/load models
    private final boolean individual;  // whether to model each variable separately
    private final int epochs;  // training epochs
    private final double learningRate;  // learning rate
    
    // prediction call counter for controlling output frequency
    private static int predictCallCount = 0;
    private static final int PREDICT_OUTPUT_INTERVAL = 100;  // output progress every N predictions
    
    // long-running Python process for prediction (avoids repeated startup)
    private Process longRunningProcess = null;
    private BufferedWriter processInput = null;
    private BufferedReader processOutput = null;
    private BufferedReader processError = null;
    private boolean processReady = false;  // whether process is ready (model loaded)
    private final Object processLock = new Object();  // lock for process access
    
    public DLinearUtil(int p, int columnCnt) {
        this(p, columnCnt, "./python/dlinear_model.py", "./models", true, 100, 0.001);
    }
    
    public DLinearUtil(int p, int columnCnt, String pythonScriptPath, String modelDir) {
        this(p, columnCnt, pythonScriptPath, modelDir, true, 10, 0.001);
    }
    
    public DLinearUtil(int p, int columnCnt, String pythonScriptPath, String modelDir, 
                      boolean individual, int epochs, double learningRate) {
        this.p = p;
        this.columnCnt = columnCnt;
        this.pythonScriptPath = pythonScriptPath;
        this.modelDir = modelDir;
        this.individual = individual;
        this.epochs = epochs;
        this.learningRate = learningRate;
        
        // ensure model directory exists
        File modelDirFile = new File(modelDir);
        if (!modelDirFile.exists()) {
            modelDirFile.mkdirs();
        }
    }
    
    /**
     * Start long-running Python process for prediction (avoids repeated startup and model load).
     */
    private void startLongRunningProcess() throws IOException {
        synchronized (processLock) {
            if (longRunningProcess != null && longRunningProcess.isAlive()) {
                return;  // process already exists
            }
            
            System.out.println("[DLinear] Starting long-running Python process for predictions...");
            
            // force arm64 architecture for Python when needed
            String osArch = System.getProperty("os.arch");
            ProcessBuilder pb;
            if (osArch.equals("aarch64") || osArch.equals("arm64")) {
                pb = new ProcessBuilder("arch", "-arm64", "python3", pythonScriptPath, "interactive", 
                        modelDir, String.valueOf(p), String.valueOf(columnCnt));
            } else {
                pb = new ProcessBuilder("python3", pythonScriptPath, "interactive", 
                        modelDir, String.valueOf(p), String.valueOf(columnCnt));
            }
            
            // set environment variables
            pb.environment().put("ARCH", "arm64");
            pb.environment().put("ARCHFLAGS", "-arch arm64");
            pb.redirectErrorStream(false);
            
            longRunningProcess = pb.start();
            processInput = new BufferedWriter(new OutputStreamWriter(longRunningProcess.getOutputStream()));
            processOutput = new BufferedReader(new InputStreamReader(longRunningProcess.getInputStream()));
            processError = new BufferedReader(new InputStreamReader(longRunningProcess.getErrorStream()));
            
            // start error stream reader thread (avoid blocking)
            Thread errorThread = new Thread(() -> {
                try {
                    String line;
                    while ((line = processError.readLine()) != null) {
                        // only print important messages (device info, errors, etc.)
                        if (line.contains("acceleration enabled") || line.contains("Device:") || 
                            line.contains("error") || line.contains("Error") || line.contains("Warning")) {
                            System.err.println("[DLinear Python] " + line);
                        }
                    }
                } catch (IOException e) {
                    // process may have closed
                }
            });
            errorThread.setDaemon(true);
            errorThread.start();
            
            // wait for process ready signal (model loaded)
            try {
                // set timeout
                long startTime = System.currentTimeMillis();
                String readySignal = null;
                while (System.currentTimeMillis() - startTime < 30000) {  // wait at most 30 seconds
                    if (processOutput.ready()) {
                        readySignal = processOutput.readLine();
                        break;
                    }
                    Thread.sleep(100);  // wait 100ms
                }
                
                if (readySignal != null && readySignal.trim().equals("READY")) {
                    processReady = true;
                    System.out.println("[DLinear] Long-running Python process ready for predictions");
                } else {
                    throw new IOException("Python process did not send READY signal. Received: " + readySignal);
                }
            } catch (IOException | InterruptedException e) {
                closeLongRunningProcess();
                throw new IOException("Failed to start long-running Python process: " + e.getMessage(), e);
            }
        }
    }
    
    /**
     * Close the long-running Python process.
     */
    private void closeLongRunningProcess() {
        synchronized (processLock) {
            processReady = false;
            
            if (processInput != null) {
                try {
                    // send exit command
                    processInput.write("EXIT\n");
                    processInput.flush();
                    processInput.close();
                } catch (IOException e) {
                    // ignore
                }
                processInput = null;
            }
            
            if (processOutput != null) {
                try {
                    processOutput.close();
                } catch (IOException e) {
                    // ignore
                }
                processOutput = null;
            }
            
            if (processError != null) {
                try {
                    processError.close();
                } catch (IOException e) {
                    // ignore
                }
                processError = null;
            }
            
            if (longRunningProcess != null) {
                longRunningProcess.destroy();
                try {
                    longRunningProcess.waitFor(2, java.util.concurrent.TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    // ignore
                }
                if (longRunningProcess.isAlive()) {
                    longRunningProcess.destroyForcibly();
                }
                longRunningProcess = null;
            }
        }
    }
    
    @Override
    public void fit(ArrayList<ArrayList<Double>> data) {
        // reset prediction counter (restart count after new training)
        synchronized (DLinearUtil.class) {
            predictCallCount = 0;
        }
        
        // delete old model files if present, so new normalization params are used
        File oldModelFile = new File(modelDir, "dlinear_model.pkl");
        if (oldModelFile.exists()) {
            System.out.println("[DLinear] Deleting old model file to ensure fresh training with normalization...");
            oldModelFile.delete();
        }
        
        System.out.println("[DLinear] Starting model training...");
        System.out.println("[DLinear] Training data size: " + data.size() + " samples, " + 
                          (data.isEmpty() ? 0 : data.get(0).size()) + " features");
        System.out.println("[DLinear] Parameters: seq_len=" + p + ", epochs=" + epochs + 
                          ", learning_rate=" + learningRate);
        
        File tempInputFile = null;
        File tempOutputFile = null;
        try {
            // create temp files for input/output
            System.out.println("[DLinear] Creating temporary files...");
            tempInputFile = File.createTempFile("dlinear_input_", ".json");
            tempOutputFile = File.createTempFile("dlinear_output_", ".json");
            tempInputFile.deleteOnExit();
            tempOutputFile.deleteOnExit();
            
            // build JSON input and write to temp file
            System.out.println("[DLinear] Writing training data to temporary file...");
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
                writer.write("  \"individual\": ");
                writer.write(String.valueOf(individual));
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
            System.out.println("[DLinear] Training data file created: " + tempInputFile.getAbsolutePath());
            
            // call Python script with temp file
            // force arm64 when needed (work around architecture mismatch)
            System.out.println("[DLinear] Calling Python script for training...");
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
            System.out.println("[DLinear] Python process started, waiting for training to complete...");
            
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
            
            // wait for reader threads to finish
            System.out.println("[DLinear] Reading Python output...");
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
                        System.out.println("[DLinear] " + line);
                    }
                }
            }
            
            if (!pythonOutput.isEmpty()) {
                // print Python progress (from stderr, shown in errorOutput)
                if (!pythonErrors.isEmpty() && !pythonErrors.contains("error")) {
                    // show training progress (normal stderr output)
                    System.out.println("[DLinear] Training Progress:");
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
                System.err.println("Python script failed (exit code: " + exitCode + ")");
                System.err.println("==========================================");
                System.err.println("stdout:");
                System.err.println(output.toString());
                System.err.println("stderr:");
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
                            throw new RuntimeException("Python script error: " + errorMsg + "\nFull output:\n" + output.toString() + "\nStderr:\n" + errorOutput.toString());
                        }
                    }
                }
                
                throw new RuntimeException("Python script failed (exit code: " + exitCode + ")\nstdout:\n" + output.toString() + "\nstderr:\n" + errorOutput.toString());
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
                System.out.println("[DLinear] Model training completed successfully!");
                
                // after training, start long-running Python process for subsequent predictions
                try {
                    startLongRunningProcess();
                } catch (IOException e) {
                    System.err.println("[DLinear] Warning: Failed to start long-running process, predictions will be slower: " + e.getMessage());
                }
            } else if (!result.isEmpty()) {
                // result not empty but no success; may be warning
                System.err.println("[DLinear] Warning: Python script output: " + result);
            }
            
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("[DLinear] Failed to call Python script: " + e.getMessage(), e);
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
        synchronized (DLinearUtil.class) {
            predictCallCount++;
        }
        
        // control output frequency: first call and every N-th call
        boolean shouldOutput = (predictCallCount == 1) || (predictCallCount % PREDICT_OUTPUT_INTERVAL == 0);
        
        if (shouldOutput && predictCallCount == 1) {
            System.out.println("[DLinear] Starting prediction phase...");
        } else if (shouldOutput) {
            System.out.println("[DLinear] Prediction progress: " + predictCallCount + " predictions completed");
        }
        
        if (window.length != p) {
            throw new IllegalArgumentException("Window size must be " + p + ", actual: " + window.length);
        }
        if (window[0].length != columnCnt) {
            throw new IllegalArgumentException("Variable count must be " + columnCnt + ", actual: " + window[0].length);
        }
        
        // try long-running process if available
        synchronized (processLock) {
            if (processReady && longRunningProcess != null && longRunningProcess.isAlive()) {
                try {
                    return predictUsingLongRunningProcess(window);
                } catch (IOException e) {
                    System.err.println("[DLinear] Warning: Long-running process failed, falling back to new process: " + e.getMessage());
                    processReady = false;
                    closeLongRunningProcess();
                    // continue with old method
                }
            }
        }
        
        // fallback: start new process each time (when long-running process unavailable)
        return predictUsingNewProcess(window);
    }
    
    /**
     * Predict using long-running process (fast path).
     */
    private ArrayList<Double> predictUsingLongRunningProcess(double[][] window) throws IOException {
        synchronized (processLock) {
            if (!processReady || longRunningProcess == null || !longRunningProcess.isAlive()) {
                throw new IOException("Long-running process not available");
            }
            
            // check if process is still alive
            if (!longRunningProcess.isAlive()) {
                processReady = false;
                throw new IOException("Python process has died");
            }
            
            // build JSON input
            StringBuilder jsonInput = new StringBuilder();
            jsonInput.append("{\n");
            jsonInput.append("  \"window\": [\n");
            for (int i = 0; i < window.length; i++) {
                jsonInput.append("    [");
                for (int j = 0; j < window[i].length; j++) {
                    jsonInput.append(window[i][j]);
                    if (j < window[i].length - 1) {
                        jsonInput.append(", ");
                    }
                }
                jsonInput.append("]");
                if (i < window.length - 1) {
                    jsonInput.append(",");
                }
                jsonInput.append("\n");
            }
            jsonInput.append("  ]\n");
            jsonInput.append("}\n");
            
            try {
                // send predict command
                processInput.write("PREDICT\n");
                processInput.write(jsonInput.toString());
                processInput.write("END\n");
                processInput.flush();
                
                // read result (with timeout)
                long startTime = System.currentTimeMillis();
                String result = null;
                while (System.currentTimeMillis() - startTime < 5000) {  // wait at most 5 seconds
                    if (processOutput.ready()) {
                        result = processOutput.readLine();
                        break;
                    }
                    Thread.sleep(10);  // wait 10ms
                }
                
                if (result == null) {
                    throw new IOException("Timeout waiting for prediction result");
                }
                
                // check if result is an error message
                if (result.contains("\"error\"")) {
                    // extract error message
                    int errorStart = result.indexOf("\"error\"");
                    if (errorStart != -1) {
                        int colonIdx = result.indexOf(":", errorStart);
                        int quoteStart = result.indexOf("\"", colonIdx);
                        int quoteEnd = result.indexOf("\"", quoteStart + 1);
                        if (quoteEnd > quoteStart) {
                            String errorMsg = result.substring(quoteStart + 1, quoteEnd);
                            throw new IOException("Python prediction error: " + errorMsg);
                        }
                    }
                    throw new IOException("Python prediction error: " + result);
                }
                
                // parse JSON output
                ArrayList<Double> prediction = parsePredictionFromJson(result);
                return prediction;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while waiting for prediction", e);
            }
        }
    }
    
    /**
     * Predict using new process (fallback method).
     */
    private ArrayList<Double> predictUsingNewProcess(double[][] window) {
        File tempInputFile = null;
        try {
            // create temp file for input
            tempInputFile = File.createTempFile("dlinear_predict_input_", ".json");
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
                writer.write("  \"individual\": ");
                writer.write(String.valueOf(individual));
                writer.write(",\n");
                writer.write("  \"model_dir\": \"");
                writer.write(modelDir);
                writer.write("\"\n");
                writer.write("}\n");
            }
            
            // call Python script with temp file
            String osArch = System.getProperty("os.arch");
            ProcessBuilder pb;
            if (osArch.equals("aarch64") || osArch.equals("arm64")) {
                pb = new ProcessBuilder("arch", "-arm64", "python3", pythonScriptPath, "predict", 
                        tempInputFile.getAbsolutePath());
            } else {
                pb = new ProcessBuilder("python3", pythonScriptPath, "predict", 
                        tempInputFile.getAbsolutePath());
            }
            pb.environment().put("ARCH", "arm64");
            pb.environment().put("ARCHFLAGS", "-arch arm64");
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
                outputThread.join(10000);
                errorThread.join(10000);
            } catch (InterruptedException e) {
                // ignore
            }
            
            if (exitCode != 0) {
                String errorJson = output.toString().trim();
                if (errorJson.isEmpty()) {
                    errorJson = errorOutput.toString().trim();
                }
                if (errorJson.contains("\"error\"")) {
                    int errorStart = errorJson.indexOf("\"error\"");
                    if (errorStart != -1) {
                        int colonIdx = errorJson.indexOf(":", errorStart);
                        int quoteStart = errorJson.indexOf("\"", colonIdx);
                        int quoteEnd = errorJson.indexOf("\"", quoteStart + 1);
                        if (quoteEnd > quoteStart) {
                            String errorMsg = errorJson.substring(quoteStart + 1, quoteEnd);
                            throw new RuntimeException("[DLinear] Python script error: " + errorMsg);
                        }
                    }
                }
                throw new RuntimeException("[DLinear] Python script execution failed (exit code: " + exitCode + ")");
            }
            
            // parse JSON output
            String result = output.toString().trim();
            ArrayList<Double> prediction = parsePredictionFromJson(result);
            return prediction;
            
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("[DLinear] Failed to call Python script: " + e.getMessage(), e);
        } finally {
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
                throw new RuntimeException("[DLinear] Cannot parse Python output, prediction field not found: " + json);
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
            throw new RuntimeException("[DLinear] Cannot parse Python output, bracket mismatch: " + json);
        }
        
        String arrayStr = cleanJson.substring(startIdx, endIdx);
        
        // handle empty array
        if (arrayStr.isEmpty()) {
            return result;
        }
        
        // split and parse numbers
        String[] values = arrayStr.split(",");
        
        for (String value : values) {
            value = value.trim();
            if (value.isEmpty()) {
                continue;
            }
            try {
                result.add(Double.parseDouble(value));
            } catch (NumberFormatException e) {
                throw new RuntimeException("[DLinear] Cannot parse number: " + value + " (from: " + json + ")", e);
            }
        }
        
        return result;
    }
    
    @Override
    public int getWindowSize() {
        return p;
    }
    
    /**
     * Clean up resources; close long-running process.
     */
    public void close() {
        closeLongRunningProcess();
    }
    
    /**
     * Finalizer; ensure process is closed.
     */
    @Override
    protected void finalize() throws Throwable {
        closeLongRunningProcess();
        super.finalize();
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
                System.err.println("[DLinear] Python script not found: " + pythonScriptPath);
                return false;
            }
            
            // check if Python is available
            System.out.println("[DLinear] Checking Python environment...");
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
                System.err.println("[DLinear] Python3 is not available");
                return false;
            }
            System.out.println("[DLinear] Python version: " + output.toString().trim());
            
            // check if PyTorch is installed
            // force arm64 for Python when needed
            String osArch = System.getProperty("os.arch");
            ProcessBuilder torchCheck;
            if (osArch.equals("aarch64") || osArch.equals("arm64")) {
                // system is arm64, use arch command to force arm64
                torchCheck = new ProcessBuilder("arch", "-arm64", "python3", "-c", "import torch; print(torch.__version__)");
            } else {
                torchCheck = new ProcessBuilder("python3", "-c", "import torch; print(torch.__version__)");
            }
            // set env for arm64
            torchCheck.environment().put("ARCH", "arm64");
            torchCheck.environment().put("ARCHFLAGS", "-arch arm64");
            // do not redirectErrorStream; read stdout and stderr separately
            Process torchProcess = torchCheck.start();
            
            // read output to avoid blocking
            StringBuilder torchOutput = new StringBuilder();
            StringBuilder torchError = new StringBuilder();
            
            // use threads to read stdout and stderr (avoid blocking)
            Thread outputThread = new Thread(() -> {
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
            
            Thread errorThread = new Thread(() -> {
                try (BufferedReader errorReader = new BufferedReader(
                        new InputStreamReader(torchProcess.getErrorStream()))) {
                    String line;
                    while ((line = errorReader.readLine()) != null) {
                        torchError.append(line).append("\n");
                    }
                } catch (IOException e) {
                    // ignore
                }
            });
            
            outputThread.start();
            errorThread.start();
            
            int torchExitCode = torchProcess.waitFor();
            
            // wait for reader threads
            try {
                outputThread.join(5000); // wait at most 5 seconds
                errorThread.join(5000);
            } catch (InterruptedException e) {
                // ignore
            }
            
            String version = torchOutput.toString().trim();
            String errorMsg = torchError.toString().trim();
            
            if (torchExitCode != 0) {
                // check for architecture mismatch
                if (errorMsg.contains("incompatible architecture") || errorMsg.contains("mach-o") || 
                    errorMsg.contains("arm64") || errorMsg.contains("x86_64")) {
                    System.err.println("[DLinear] Warning: PyTorch architecture mismatch!");
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
                    System.err.println("Note: Program will continue, but DLinear model may not work properly.");
                } else {
                    System.err.println("[DLinear] Warning: PyTorch not installed or import failed, run: pip install -r python/requirements.txt");
                    if (!errorMsg.isEmpty()) {
                        System.err.println("[DLinear] Error message: " + errorMsg);
                    }
                }
            } else {
                // PyTorch installed; show version
                if (!version.isEmpty()) {
                    System.out.println("[DLinear] PyTorch installed: " + version);
                } else if (!errorMsg.isEmpty()) {
                    // stdout empty but stderr has content; may be warning
                    System.out.println("[DLinear] PyTorch check completed (may have warnings)");
                }
            }
            
            // return true even if PyTorch check failed, so program can continue
            // architecture mismatch may be only a warning; runtime may still work
            return true;
        } catch (Exception e) {
            System.err.println("[DLinear] Failed to check Python environment: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
}
