package de.saar.minecraft.analysis;

import com.google.gson.Gson;
import com.google.gson.JsonParser;
import de.bwaldvogel.liblinear.*;
import de.saar.basic.Pair;

import de.up.ling.tree.Tree;
import de.up.ling.tree.TreeVisitor;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.stat.descriptive.UnivariateStatistic;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jooq.DSLContext;
import org.jooq.exception.DataAccessException;
import org.jooq.impl.DSL;

import java.sql.SQLException;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

import picocli.CommandLine;

import static de.saar.minecraft.broker.db.Tables.GAMES;
import static de.saar.minecraft.broker.db.Tables.GAME_LOGS;


public class WeightEstimator {
    private static final Logger logger = LogManager.getLogger(WeightEstimator.class);

    public static final String FIRST_INSTRUCTION_FEATURE = "firstinstruction";
    
    protected static final String FIRST_OCCURENCE_PREFIX = "firstoccurence_";

    public Map<String, Integer> featureMap;
    public int maxFeatureId;
    
    public final int lowerPercentile;
    public final int higherPercentile;

    /**
     * A list containing timing data for each game. The data for a game is again a list, this
     * time of (feature array, time) pairs.
     */
    public List<List<Pair<List<String>, Long>>> allData;
    
    private final DSLContext jooq;
    
    public static class WeightResult {
        public final Map<String, Double> weights;
        public final Map<String, Double> firstOccurenceWeights;

        public WeightResult(Map<String, Double> allWeights) {
            this.weights = new HashMap<>();
            this.firstOccurenceWeights = new HashMap<>();
            for (var e: allWeights.entrySet()) {
                if (e.getKey().startsWith(FIRST_OCCURENCE_PREFIX)) {
                    firstOccurenceWeights.put(e.getKey().substring(FIRST_OCCURENCE_PREFIX.length()), e.getValue());
                } else {
                    weights.put(e.getKey(), e.getValue());
                }
            }
        }
        
        public static WeightResult fromJson(String jsonString) {
            Gson gson = new Gson();
            return gson.fromJson(jsonString, WeightResult.class);
        }
     
        private String weightMapToString(Map<String, Double> map) {
            return map.entrySet()
                    .stream()
                    .sorted(Map.Entry.comparingByKey())
                    .map((entry) -> "\"" + entry.getKey() + "\": " + + entry.getValue())
                    .collect(Collectors.joining(",\n"));
        }
        
        public String toJson() {
            StringBuilder sb = new StringBuilder();
            sb.append("{ \"firstOccurenceWeights\": {\n");
            sb.append(weightMapToString(firstOccurenceWeights));
            sb.append("\n},\n\"weights\": {\n");
            sb.append(weightMapToString(weights));
            sb.append("\n}}");
            return sb.toString();
        }
        
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("first occurence weights:\n");
            firstOccurenceWeights.entrySet().stream().sorted(Map.Entry.comparingByKey()).forEach((entry) ->
                    sb.append(entry.getKey())
                            .append(": ")
                            .append(entry.getValue())
                            .append("\n")
            );
            sb.append("weights:\n");
            weights.entrySet().stream().sorted(Map.Entry.comparingByKey()).forEach((entry) ->
                   sb.append(entry.getKey())
                           .append(": ")
                           .append(entry.getValue())
                           .append("\n")
            );
            return sb.toString();
        }
    }

    public static void main(String[] args) throws SQLException {
        new CommandLine(new WeightEstimatorCLI()).execute(args);
    }

    /**
     * Creates an estimator that will sample from bootstrapped coefficients between {@code lowerPercentile} and
     * {@code higherPercentile}.
     * @param connStr The jdbc / jooq connection string.  Jooq ignores user and pass info in this string.
     * @param user The database username
     * @param pass the database password
     * @param lowerPercentile coefficients are uniformly drawn from this percentile of the bootstrapped coefficients
     * @param higherPercentile coefficients are randomly drawn up to this percentile of the bootstrapped coefficients
     */
    public WeightEstimator(String connStr, String user, String pass, int lowerPercentile, int higherPercentile,
                           List<List<Tree<String>>> seedGameData) {
        this(connStr, user, pass, lowerPercentile, higherPercentile, seedGameData, "");
    }

    /**
     * Creates an estimator that will sample from bootstrapped coefficients between {@code lowerPercentile} and
     * {@code higherPercentile}.
     * @param connStr The jdbc / jooq connection string.  Jooq ignores user and pass info in this string.
     * @param user The database username
     * @param pass the database password
     * @param lowerPercentile coefficients are uniformly drawn from this percentile of the bootstrapped coefficients
     * @param higherPercentile coefficients are randomly drawn up to this percentile of the bootstrapped coefficients
     * @param architect only use games played with this architect. if empty, use all games
     */
    public WeightEstimator(String connStr, String user, String pass, int lowerPercentile, int higherPercentile,
                           List<List<Tree<String>>> seedGameData, String architect) {
        this.jooq = DSL.using(connStr, user, pass);
        this.lowerPercentile = lowerPercentile;
        this.higherPercentile = higherPercentile;
        try {
            if (architect.equals("")) {
                this.allData = extractAllData();
            } else {
                this.allData = extractDataForArchitect(architect);
            }
            if (allData.stream().mapToInt((List::size)).sum() == 0) {
                // there are games but they have no instructions yet
                // in other parts of the code we just check whether
                // allData is empty instead of also checking whether each
                // list is empty, so set allData to the empty list here
                this.allData = new ArrayList<>();
            }
        } catch (DataAccessException e) {
            this.allData = new ArrayList<>();
        }
        logger.info("initialized with " + allData.size() + " games plus " + seedGameData.size() + " seed games" +
                " for architect " + architect);
        addSeedGamesToAllData(seedGameData);
        createFeatureMapping(allData);
    }

    private void addSeedGamesToAllData(List<List<Tree<String>>> games) {
        long assumedGameTime = 1000*10; // 10 seconds in milliseconds, hopefully faster than real completion times
        var seedInstructionTimes = new ArrayList<Pair<List<String>, Double>>();
        for (var game: games) {
            var gameTimes = new ArrayList<Pair<List<String>, Long>>();
            int numInstructions = game.size();
            for (var instruction: game) {
                ArrayList<String> labels = new ArrayList<>();
                instruction.dfs(new TreeVisitor<String, Void, Void>() {
                    @Override
                    public Void combine(Tree<String> node, List<Void> childrenValues) {
                        labels.add(node.getLabel());
                        return null;
                    }
                });
                gameTimes.add(new Pair<>(labels, assumedGameTime / numInstructions));
            }
            allData.add(gameTimes);
        }
    }

    /**
     * Runs L2 regression on the data.  Returns a vector of coefficients.
     */
    protected double[] runLinearRegression(List<Pair<List<String>, Long>> data) {
        // for some reason we were asked to solve an empty problem
        // return a low-ball estimate instead.
        if (data.isEmpty()) {
            var result = new double[maxFeatureId];
            Arrays.fill(result, 0.9);
            return result;
        }
        var p = new Problem();
        var pdata = dataToLibLinearFormat(data);
        // p.bias = -1; // this is the default value
        p.x = pdata.left;
        p.y = pdata.right;
        p.l = pdata.left.length;
        p.n = maxFeatureId;
        Parameter params = new Parameter(SolverType.L2R_L2LOSS_SVR, 500, 0.01);
        Linear.disableDebugOutput();
        Model m = Linear.train(p, params);
        return m.getFeatureWeights();
    }

    /**
     * Use the feature mapping to create a map that can be used for MinecraftRealizer:SetExpectedDurations.
     */
    protected Map<String, Double> linearRegressionResultToGrammarDurations(double[] coeffs) {
        Map<String, Double> result = new HashMap<>();
        for (var entry: featureMap.entrySet()) {
            result.put(entry.getKey(), coeffs[entry.getValue()-1]);
        }
        return result;
    }

    /**
     * Extracts timing data from all games in the database and optimizes weights using linear regression.
     */
    public WeightResult predictDurationCoeffsFromAllGames() {
        if (allData.isEmpty()) {
            return new WeightResult(new HashMap<>());
        }
        var flatData = allData.stream().reduce(new ArrayList<>(), (x, y) -> { x.addAll(y); return x;});
        var coeffs = runLinearRegression(flatData);
        return new WeightResult(linearRegressionResultToGrammarDurations(coeffs));
    }

    /**
     * Runs Bootstrapping for {@code numRuns} iterations on the set of all instructions,
     * running linear regression on each sample.  We then sample each weight between the {@code lowerPercentile} and
     * the {@code higherPercentile} of all regression results for that feature.  Sampling is uniform.
     * @param numRuns humber of runs for bootstrapping
     * @return A map of feature weights
     */
    public WeightResult sampleDurationCoeffsWithBootstrap(int numRuns, boolean samplePerGame) {
        if (allData.isEmpty()) {
            return new WeightResult(new HashMap<>());
        }
        double[][] bootstrapData;
        if (samplePerGame) {
            bootstrapData = perGameBootstrap(numRuns);
        } else {
            bootstrapData = perInstructionBootstrap(numRuns);
        }
        var bootResult = statisticsFromBootstrap(bootstrapData);
        var r = new Random();
        var result = new HashMap<String, Double>();
        for (var entry: featureMap.entrySet()) {
            double from = bootResult.lowerbound[entry.getValue() - 1];
            double to =  bootResult.upperbound[entry.getValue() - 1];
            result.put(entry.getKey(), r.nextDouble()*(to-from) + from);
        }
        return new WeightResult(result);
    }

    /**
     * Bootstrapping with game-based sampling, i.e. one draw adds all instructions of one random
     * game into the sample for which we run linear regression.
     * Other than the different sampling, it does the same as {@link #perInstructionBootstrap(int)}
     * @param numRuns number of runs for bootstrapping
     * @return A map of feature weights
     */
    protected double[][] perGameBootstrap(int numRuns) {
        int n = allData.size();
        var random = new Random();
        random.setSeed(1L);
        var results = new double[numRuns][];
        for (int run = 0; run < numRuns; run++) {
            var sample = new ArrayList<Pair<List<String>, Long>>();
            for (int i = 0; i < n; i++) {
                sample.addAll(allData.get(random.nextInt(n)));
            }
            results[run] = runLinearRegression(sample);
        }
        return results;
    }
    
    /**
     * Run Bootstrapping for {@code numRuns} iterations. For each sample, run regression on the feature weights.
     * @return a vector of size{@code [numRuns][numFeatures]} containing all coefficients for every run.
     * Feature indices are the ones from the feature mapping.
     */
    protected double[][] perInstructionBootstrap(int numRuns) {
        var flatData = allData.stream().reduce(new ArrayList<>(), (x, y) -> { x.addAll(y); return x;});
        int n = flatData.size();
        var random = new Random();
        random.setSeed(1L);
        var results = new double[numRuns][];
        for (int run = 0; run < numRuns; run++) {
            var sample = new ArrayList<Pair<List<String>, Long>>();
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
    protected BootstrapResult statisticsFromBootstrap(double[][] bootstrapResult) {
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

            logger.debug("==========================================");
            for (var x: featureMap.entrySet()) {
                if (feature == x.getValue() - 1) {
                    logger.debug(x.getKey());
                }
            }
            logger.debug("mean: " + mean.evaluate(dataArray));
            logger.debug(lowerPercentile + "%: " + lowerBound.evaluate(dataArray));
            logger.debug(higherPercentile + "%: " + upperBound.evaluate(dataArray));
        }
        return result;
    }

    /**
     * Extracts timing data from the database that were played with a specific architect.
     * @return A list of games, each being a list of timings.
     */
    private List<List<Pair<List<String>, Long>>> extractDataForArchitect(String architect) {
        return jooq.select(GAMES.ID)
                .from(GAMES)
                .where(GAMES.ARCHITECT_INFO.eq(architect))
                .fetch(GAMES.ID)
                .stream()
                .map(this::extractDataFromGame)
                // some games have no instructions, especially the one just started,
                // which is nonetheless already recorded in the DB with no instructions so far.
                // remove them so we cannot create a bootstrap without instructions.
                .filter((x) -> ! x.isEmpty())
                .collect(Collectors.toList());
    }
    
    /**
     * Extracts timing data from the database.
     * @return A list of games, each being a list of timings.
     */
    private List<List<Pair<List<String>, Long>>> extractAllData() {
        return jooq.select(GAMES.ID)
                .from(GAMES)
                .fetch(GAMES.ID)
                .stream()
                .map(this::extractDataFromGame)
                // some games have no instructions, especially the one just started,
                // which is nonetheless already recorded in the DB with no instructions so far.
                // remove them so we cannot create a bootstrap without instructions.
                .filter((x) -> ! x.isEmpty())
                .collect(Collectors.toList());
    }
    
    private List<Pair<List<String>, Long>> extractDataFromGame(int gameId) {
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
                // ignore everything after the end of the experiment
                .takeWhile((x) -> ! x.getLeft().contains("Thank you for participating in our experiment"))
                .filter((x) -> x.left.startsWith("{"))
                .map((x) -> new Pair<>(JsonParser.parseString(x.left).getAsJsonObject(), x.right))
                .collect(Collectors.toList());

        List<Pair<List<String>, Long>> result = new ArrayList<>();
        
        Set<String> seenIndefiniteObjects = new HashSet<>();
        
        List<String> lastInstruction = null;
        LocalDateTime lastTime = null;
        
        for (var instructionTime: instructionTimes) {
            var json = instructionTime.left;
            if (! (json.has("new") && json.get("new").getAsBoolean())) {
                continue;
            }
            String instructionTree = json.get("tree").getAsString();
            if (instructionTree.equals("NULL")) {
                String message = json.get("message").getAsString();
                if (message.startsWith("Now I will teach you how to build a ")) {
                    // the user now knows this concept through teaching
                    seenIndefiniteObjects.add(message.substring("Now I will teach you how to build a ".length()));
                }
                // this is an instruction such as "now I will teach you how to build a wall"
                continue;
            }
            List<String> nonProductiveTerminalSymbols = List.of("dnp", "np", "obj", "loc");
            ArrayList<String> currInstruction = Arrays.stream(instructionTree.split("[(),]+"))
                    .filter((x) -> !nonProductiveTerminalSymbols.contains(x))
                    .collect(Collectors.toCollection(ArrayList::new));
            // currInstruction;
            // Arrays.stream(currInstruction).filter((x) -> true).toArray()
            // mark all indefinite objects we have not yet seen
            for (int i = 0; i < currInstruction.size(); i++) {
                String f = currInstruction.get(i);
                if (f.startsWith("i") && ! seenIndefiniteObjects.contains(f)) {
                    seenIndefiniteObjects.add(f);
                    currInstruction.set(i, FIRST_OCCURENCE_PREFIX + f);
                }
            }
            if (lastTime == null) { // first instruction
                lastTime = instructionTime.right;
                lastInstruction = currInstruction;
                lastInstruction.add(FIRST_INSTRUCTION_FEATURE);
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
    private void createFeatureMapping(Collection<List<Pair<List<String>, Long>>> data) {
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
    private Pair<Feature[][], double[]> dataToLibLinearFormat(List<Pair<List<String>, Long>> data) {
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
}
