package de.saar.minecraft.analysis;

import com.google.gson.JsonParser;
import de.bwaldvogel.liblinear.*;
import de.saar.basic.Pair;
// import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.stat.descriptive.UnivariateStatistic;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.jooq.DSLContext;
import org.jooq.impl.DSL;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

import static de.saar.minecraft.broker.db.Tables.GAMES;
import static de.saar.minecraft.broker.db.Tables.GAME_LOGS;

public class WeightEstimator {
    /*
    TODO:
    implement block bootstrap with block == games (maybe?)
     */

    public static final String FIRST_INSTRUCTION_FEATURE = "firstinstruction";

    public Map<String, Integer> featureMap;
    public int maxFeatureId;
    
    public final int lowerPercentile;
    public final int higherPercentile;
    
    public List<List<Pair<String[], Long>>> allData;
    
    private final DSLContext jooq;
    
    public static void main(String[] args) throws SQLException {
        var connStr = "jdbc:mariadb://localhost:3306/RANDOMIZEDWEIGHTS";
        if (args.length >= 1) {
            connStr = args[0];
        }
        String connUser = "minecraft";
        if (args.length >= 2) {
            connUser = args[1];
        }
        String connPwd ="";
        if (args.length >= 3) {
            connPwd = args[2];
        }
        Connection conn = DriverManager.getConnection(connStr, connUser, connPwd);

        var estimator = new WeightEstimator(DSL.using(conn),25,75);
        var results = estimator.predictDurationCoeffsFromAllGames();
        System.out.println("global optimum:");
        printWeightMap(results);

        System.out.println("sampled:");
        ArrayList<Map<String, Double>> res = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            res.add(estimator.sampleDurationCoeffsWithBootstrap(1000));
        }
        for (var key: estimator.featureMap.keySet()) {
            System.out.print(key + ": ");
            for (int i = 0; i < 10; i++) {
                System.out.print(res.get(i).get(key) + " ");
            }
            System.out.println();
        }
        // estimator.sampleDurationCoeffsWithBootstrap(10000);
    }
    
    public WeightEstimator(String connStr, int lowerPercentile, int higherPercentile) {
        this(DSL.using(connStr), lowerPercentile, higherPercentile);
    }
    
    public WeightEstimator(DSLContext jooq, int lowerPercentile, int higherPercentile) {
        this.jooq = jooq;
        this.lowerPercentile = lowerPercentile;
        this.higherPercentile = higherPercentile;
        this.allData = extractAllData();
        createFeatureMapping(allData);
    }

    /**
     * Runs L2 regression on the data.  Returns a vector of coefficients.
     */
    public double[] runLinearRegression(List<Pair<String[], Long>> data) {
        var p = new Problem();
        var pdata = dataToLibLinearFormat(data);
        // p.bias = -1; // this is the default value
        p.x = pdata.left;
        p.y = pdata.right;
        p.l = pdata.left.length;
        p.n = maxFeatureId;
        Parameter params = new Parameter(SolverType.L2R_L2LOSS_SVR, 500, 0.01);
        Model m = Linear.train(p, params);
        return m.getFeatureWeights();
    }

    /**
     * Use the feature mapping to create a map that can be used for MinecraftRealizer:SetExpectedDurations.
     */
    public Map<String, Double> linearRegressionResultToGrammarDurations(double[] coeffs) {
        Map<String, Double> result = new HashMap<>();
        for (var entry: featureMap.entrySet()) {
            result.put(entry.getKey(), coeffs[entry.getValue()-1]);
        }
        return result;
    }

    /**
     * Extracts timing data from all games in the database and optimizes weights using linear regression.
     */
    public Map<String, Double> predictDurationCoeffsFromAllGames() {
        var flatData = allData.stream().reduce(new ArrayList<>(), (x, y) -> { x.addAll(y); return x;});
        var coeffs = runLinearRegression(flatData);
        return linearRegressionResultToGrammarDurations(coeffs);
    }

    /**
     * Runs Bootstrapping for {@code numRuns} iterations on the set of all instructions,
     * running linear regression on each sample.  We then sample each weight between the {@code lowerPercentile} and
     * the {@code higherPercentile} of all regression results for that feature.  Sampling is uniform.
     * @param numRuns humber of runs for bootstrapping
     * @return A map of feature weights
     */
    public Map<String, Double> sampleDurationCoeffsWithBootstrap(int numRuns) {
        double[][] bootstrapData = perElementBootstrap(numRuns);
        var bootResult = statisticsFromBootstrap(bootstrapData);
        var r = new Random();
        var result = new HashMap<String, Double>();
        for (var entry: featureMap.entrySet()) {
            double from = bootResult.lowerbound[entry.getValue() - 1];
            double to =  bootResult.upperbound[entry.getValue() - 1];
            result.put(entry.getKey(), r.nextDouble()*(to-from) + from);
        }
        return result;
    }

    /**
     * Run Bootstrapping for {@code numRuns} iterations. For each sample, run regression on the feature weights.
     * @return a vector of size{@code [numRuns][numFeatures]} containing all coefficients for every run.
     * Feature indices are the ones from the feature mapping.
     */
    protected double[][] perElementBootstrap(int numRuns) {
        var flatData = allData.stream().reduce(new ArrayList<>(), (x, y) -> { x.addAll(y); return x;});
        int n = flatData.size();
        var random = new Random();
        random.setSeed(1L);
        var results = new double[numRuns][];
        for (int run = 0; run < numRuns; run++) {
            var sample = new ArrayList<Pair<String[], Long>>();
            for (int i = 0; i < n; i++) {
                sample.add(flatData.get(random.nextInt(n)));
            }
            results[run] = runLinearRegression(sample);
        }
        return results;
    }

    private static class BootstrapResult {
        double[] means;
        double[] lowerbound;
        double[] upperbound;

        public BootstrapResult(int numFeatures) {
            this.means = new double[numFeatures];
            this.lowerbound = new double[numFeatures];
            this.upperbound = new double[numFeatures];
        }
    }

    /**
     * Computes the mean, lower percentile and upper percentile for each feature present in the {@code bootstrapResult}.
     * @return A populated {@link BootstrapResult}
     */
    public BootstrapResult statisticsFromBootstrap(double[][] bootstrapResult) {
        UnivariateStatistic lowerBound = new Percentile(lowerPercentile);
        UnivariateStatistic upperBound = new Percentile(higherPercentile);
        UnivariateStatistic mean = new Mean();
        
        int numFeatures = bootstrapResult[0].length;
        int numRuns = bootstrapResult.length;
        BootstrapResult result = new BootstrapResult(numFeatures);

        for (int feature = 0; feature < numFeatures; feature++) {
            SummaryStatistics statistics = new SummaryStatistics();
            double[] dataArray = new double[numRuns];
            for (int run = 0; run < numRuns; run++) {
                dataArray[run] = bootstrapResult[run][feature];
                statistics.addValue(bootstrapResult[run][feature]);
            }
            result.means[feature] = mean.evaluate(dataArray);
            result.lowerbound[feature] = lowerBound.evaluate(dataArray);
            result.upperbound[feature] = upperBound.evaluate(dataArray);

            System.out.println("==========================================");
            for (var x: featureMap.entrySet()) {
                if (feature == x.getValue() - 1) {
                    System.out.println(x.getKey());
                }
            }
            System.out.println("mean: " + mean.evaluate(dataArray));
            System.out.println(lowerPercentile + "%: " + lowerBound.evaluate(dataArray));
            System.out.println(higherPercentile + "%: " + upperBound.evaluate(dataArray));
        }
        return result;
    }
    
    private List<List<Pair<String[], Long>>> extractAllData() {
        return jooq.select(GAMES.ID)
                .from(GAMES)
                .fetch(GAMES.ID)
                .stream()
                .map(this::extractDataFromGame)
                .collect(Collectors.toList());
    }
    
    private List<Pair<String[], Long>> extractDataFromGame(int gameId) {
        var instructionTimes = jooq.selectFrom(GAME_LOGS)
                .where(GAME_LOGS.GAMEID.eq(gameId))
                .and(GAME_LOGS.MESSAGE_TYPE.eq("TextMessage"))
                .stream()
                .map((x) -> {
                    var text = JsonParser.parseString(x.getMessage())
                            .getAsJsonObject()
                            .get("text")
                            .getAsString();
                    return new Pair<>(text, x.getTimestamp());
                })
                .filter((x) -> x.left.startsWith("{"))
                .map((x) -> new Pair<>(JsonParser.parseString(x.left).getAsJsonObject(), x.right))
                .collect(Collectors.toList());

        List<Pair<String[], Long>> result = new ArrayList<>();
        
        String[] lastInstruction = null;
        LocalDateTime lastTime = null;
        
        for (var instructionTime: instructionTimes) {
            var json = instructionTime.left;
            if (! (json.has("new") && json.get("new").getAsBoolean())) {
                continue;
            }
            String instructionTree = json.get("tree").getAsString();
            if (instructionTree.equals("NULL")) {
                // this is an instruction such as "now I will teach you how to build a wall"
                continue;
            }
            var currInstruction = instructionTree.split("[(),]+");
            if (lastTime == null) { // first instruction
                lastTime = instructionTime.right;
                lastInstruction = new String[currInstruction.length + 1];
                System.arraycopy(currInstruction, 0, lastInstruction, 0, currInstruction.length);
                lastInstruction[currInstruction.length] = FIRST_INSTRUCTION_FEATURE;
                continue;
            }
            result.add(new Pair<>(lastInstruction, lastTime.until(instructionTime.right, ChronoUnit.MILLIS)));
            lastInstruction = currInstruction;
            lastTime = instructionTime.right;
        }

        var successTimeR = jooq.select(GAME_LOGS.TIMESTAMP)
                .from(GAME_LOGS)
                .where(GAME_LOGS.GAMEID.eq(gameId))
                .and(GAME_LOGS.MESSAGE.contains("SuccessfullyFinished"))
                .fetchOne();
        
        if (successTimeR != null && lastTime != null) {
            result.add(new Pair<>(lastInstruction, lastTime.until(successTimeR.component1(), ChronoUnit.MILLIS)));
        }
        return result;
    }


    /**
     * initializes the feature mapping from the given games.
     */
    private void createFeatureMapping(Collection<List<Pair<String[], Long>>> data) {
        Map<String, Integer> result = new HashMap<>();
        int currentId = 1;
        for (var game: data) {
            for (var instruction: game) {
                for (var elem: instruction.left) {
                    if (result.putIfAbsent(elem, currentId) == null) {
                        currentId++;
                    }
                }
            }
        }
        featureMap = result;
        maxFeatureId = currentId - 1;
    }

    /**
     * Converts a list of String[], Long pairs from our data model to a X,Y representation usable
     * by liblinear.
     */
    private Pair<Feature[][], double[]> dataToLibLinearFormat(List<Pair<String[], Long>> data) {
        var x = new Feature[data.size()][];
        var y = new double[data.size()];
        int index = 0;
        for (var instruction: data) {
            int[] featureCounts = new int[this.maxFeatureId+1];
            for (var feat: instruction.left) {
                featureCounts[featureMap.get(feat)] += 1;
            }
            var features = new ArrayList<FeatureNode>();
            for (int i = 0; i <= maxFeatureId; i++) {
                if (featureCounts[i] > 0) {
                    features.add(new FeatureNode(i, featureCounts[i]));
                }
            }
            x[index] = features.toArray(new Feature[0]);
            y[index] = instruction.right;
            index++;
        }
        return new Pair<>(x,y);
    }
    
    private static void printWeightMap(Map<String,Double> map) {
        map.entrySet().stream().sorted(Comparator.comparing(Map.Entry::getKey)).forEach((entry) ->
                System.out.println(entry.getKey() + ": " + entry.getValue())
        );
    }
}
