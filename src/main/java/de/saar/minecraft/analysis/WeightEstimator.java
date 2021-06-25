package de.saar.minecraft.analysis;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
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

    /** This is a debug feature: set to true, every tree is treated as *one* feature
        i.e. we learn one weight per instruction and have no weight sharing between
        instructions via features.  Obviously does not generalize at all.  Used to
        get an intuition for the amount of noise in the data that we cannot remove
        by better features / modeling. */
    public boolean singleFeaturePerGame = false;

    public final boolean deletionsAsCost;
    
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

    public static void main(String[] args) {
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
                           List<List<Tree<String>>> seedGameData, boolean deletionsAsCost) {
        this(connStr, user, pass, lowerPercentile, higherPercentile, seedGameData, "", deletionsAsCost);
    }

    public WeightEstimator(String connStr, String user, String pass, int lowerPercentile, int higherPercentile,
                           List<List<Tree<String>>> seedGameData, String architect, boolean deletionsAsCost) {
        this(connStr, user, pass, lowerPercentile, higherPercentile, seedGameData, architect, "%", deletionsAsCost);
    }
    /**
     * Creates an estimator that will sample from bootstrapped coefficients between {@code lowerPercentile} and
     * {@code higherPercentile}.
     * @param connStr The jdbc / jooq connection string.  Jooq ignores user and pass info in this string.
     * @param user The database username
     * @param pass the database password
     * @param lowerPercentile coefficients are uniformly drawn from this percentile of the bootstrapped coefficients
     * @param higherPercentile coefficients are randomly drawn up to this percentile of the bootstrapped coefficients
     * @param architect only use games played with this architect. if empty, use all games. Can use % and _ to match
     *                  architects using SQL LIKE.
     */
    public WeightEstimator(String connStr, String user, String pass, int lowerPercentile, int higherPercentile,
                           List<List<Tree<String>>> seedGameData, String architect, String scenario, boolean deletionsAsCost) {
        this.jooq = DSL.using(connStr, user, pass);
        this.lowerPercentile = lowerPercentile;
        this.higherPercentile = higherPercentile;
        this.deletionsAsCost = deletionsAsCost;
        if (architect.equals("")) {
            architect = "%";
        }
        if (scenario.equals("")) {
            // match everything
            scenario = "%";
        }
        try {
            this.allData = extractData(architect, scenario);
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

    public WeightResult getUCBWithBootstrap(int numRuns, boolean samplePerGame) {
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
        var result = new HashMap<String, Double>();
        for (var entry: featureMap.entrySet()) {
            double from = bootResult.lowerbound[entry.getValue() - 1];
            result.put(entry.getKey(), from);
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
     * Extracts timing data from the database that were played with an architect
     * matching the supplied architectMatch. architectMatch is matched against the
     * architect names using SQL LIKE, i.e. "foo" matches only "foo" while "foo%"
     * also matches games played with an architect called "foobar".
     * @return A list of games, each being a list of timings.
     */
    private List<List<Pair<List<String>, Long>>> extractData(String architectMatch, String scenarioMatch) {
        return jooq.select(GAMES.ID)
                .from(GAMES)
                .where(GAMES.ARCHITECT_INFO.like(architectMatch))
                .and(GAMES.SCENARIO.like(scenarioMatch))
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

    private record InstructionMetaData(JsonObject object,
                                       JsonObject json,
                                       LocalDateTime tstamp,
                                       int deletionsUpToNow) {}

    /**
     * converts a string to a comparable json object, i.e. it removes the "blocks left" property
     * that can change over time.
     */
    private JsonObject objectStringToGeneralObject(String obj) {
        var res = JsonParser.parseString(obj).getAsJsonObject();
        res.remove("blocks");
        return res;
    }

    private List<Pair<List<String>, Long>> extractDataFromGame(int gameId) {
        return extractDataFromGame(gameId, false);
    }
        /**
         * Extracts (instruction, cost) tuples from a specific game.  The cost is either
         * the time it took to complete an instruction or the number of errors for that instruction.
         * @param gameId the ID of the game
         * @return List of (instruction, cost) pairs.
         */
    private List<Pair<List<String>, Long>> extractDataFromGame(int gameId, boolean costIsNumErrors) {
        /*
        This code is a bit involved.  We want a list of instructions we chose to give, but the actual log
        also contains correction instructions, e.g. when a block was deleted and needs to be instructed.
        The log contains three types of information:
         - text messages are the instructions we give to the user.
         - current object is the object we consider.  It appears before the instruction.
         - block destruction messages are counted because we might want to use this as our cost metric.

         In a first step, we record each instruction and at which time it was started and how many deletions
         occured up to that point.  In the second step, we calculate the differences between times (or deletions)
         to obtain the cost of each instruction.
         */
        var log = jooq.selectFrom(GAME_LOGS)
                .where(GAME_LOGS.GAMEID.eq(gameId))
                .and(
                        GAME_LOGS.MESSAGE_TYPE.eq("TextMessage")
                                .or(GAME_LOGS.MESSAGE_TYPE.eq("CurrentObject"))
                                .or(GAME_LOGS.MESSAGE_TYPE.eq("BlockDestroyedMessage"))
                )
                .fetch();
        List<InstructionMetaData> instructionTimes = new ArrayList<>();
        JsonObject currentObject = null;
        int deletionsUpToNow = 0;
        /*
        first step
         */
        for (var elem: log) {
            if (elem.getMessageType().equals("BlockDestroyedMessage")) {
                deletionsUpToNow += 1;
                continue;
            }
            String message = elem.get(GAME_LOGS.MESSAGE);
            String type = elem.get(GAME_LOGS.MESSAGE_TYPE);
            if (message.contains("You finished building a")) {
                // this is meta-instruction we can completely ignore; the relevant meta-instruction
                // we use is the start teaching instruction instead to note when a new concept is introduced.
                continue;
            }
            if (message.contains("Thank you for participating in our experiment")) {
                break; // end of game
            }
            if (type.equals("CurrentObject")) {
                currentObject = objectStringToGeneralObject(message);
                /* the following loop checks whether we revisit an old object to be instructed.
                   this happens when a block is deleted and has to be instructed.  We want
                   to count the initial instruction (e.g. "build a railing"), the correction instruction,
                   and finishing the instruction all as part of the initial instruction.  Therefore,
                   if we find the new currentObject in the list of objects, we turn back to the state
                   before that object was first introduced.
                   As only high-level objects can be continued, we skip this for blocks.
                 */
                String currentType = currentObject.get("type").getAsString();
                if (! currentType.equals("Block")) {
                    for (int i = 0; i < instructionTimes.size(); i++) {
                        if (instructionTimes.get(i).object == null) {
                            continue;
                        }
                        var candidateObject = instructionTimes.get(i).object;
                        if (candidateObject.equals(currentObject)) {
                            instructionTimes = new ArrayList<>(instructionTimes.subList(0, i + 1));
                            break;
                        }
                    }
                }
                continue;
            }
            JsonObject finalCurrentObject = currentObject;
            if (instructionTimes.stream().anyMatch((x) -> x.object!= null && x.object.equals(finalCurrentObject))) {
                // We already started tracking this element and do not want to add a second item
                // for it into our list.  This happens when we go back to a high-level instruction (see above)
                continue;
            }
            // Message type: TextMessage
            var text = JsonParser.parseString(elem.getMessage())
                    .getAsJsonObject()
                    .get("text")
                    .getAsString();
            if (! text.startsWith("{")) {
                continue;
            }
            var jsonObject = JsonParser.parseString(text).getAsJsonObject();
            if (! (jsonObject.has("new") && jsonObject.get("new").getAsBoolean())) {
                // this is a repetition of a previous instruction; no need to store it
                continue;
            }
            if (text.contains("I will teach you how")) {
                // this is a meta instruction where we do not actually instruct a specific object, so
                // we save null as the object.
                instructionTimes.add(new InstructionMetaData(null, jsonObject, elem.getTimestamp(), deletionsUpToNow));
            } else {
                instructionTimes.add(new InstructionMetaData(currentObject, jsonObject, elem.getTimestamp(), deletionsUpToNow));
            }
        }

        /*
        second step
         */
        Set<String> seenIndefiniteObjects = new HashSet<>();
        List<String> lastInstruction = null;
        LocalDateTime lastTime = null;
        int lastNumDeletions = 0;
        List<Pair<List<String>, Long>> result = new ArrayList<>();

        for (var metaData: instructionTimes) {
            var json = metaData.json;
            String instructionTree = json.get("tree").getAsString();
            if (instructionTree.equals("NULL")) {
                String message = json.get("message").getAsString();
                if (message.startsWith("Now I will teach you how to build a ")) {
                    // the user now knows this concept through teaching
                    seenIndefiniteObjects.add("i" + message.substring("Now I will teach you how to build a ".length()));
                }
                // this is an instruction such as "now I will teach you how to build a wall"
                continue;
            }
            ArrayList<String> currInstruction = Arrays.stream(instructionTree.split("[(),]+"))
                    .collect(Collectors.toCollection(ArrayList::new));
            // mark all indefinite objects we have not yet seen
            boolean newInstructionFound = false;
            for (int i = 0; i < currInstruction.size(); i++) {
                String f = currInstruction.get(i);
                if (f.startsWith("i") && ! seenIndefiniteObjects.contains(f)) {
                    seenIndefiniteObjects.add(f);
                    currInstruction.set(i, FIRST_OCCURENCE_PREFIX + f);
                    newInstructionFound = true;
                }
            }
            
            if (singleFeaturePerGame) {
                ArrayList<String> tmp = new ArrayList<>();
                if (newInstructionFound) {
                    tmp.add(instructionTree + "FIRSTINSTRUCTION");
                } else {
                    tmp.add(instructionTree);
                }
                currInstruction = tmp;
            }
            
            if (lastTime == null) { // first instruction
                lastTime = metaData.tstamp;
                lastInstruction = currInstruction;
                lastInstruction.add(FIRST_INSTRUCTION_FEATURE);
                continue;
            }
            long cost;
            if (costIsNumErrors) {
                cost = metaData.deletionsUpToNow - lastNumDeletions;
            } else {
                cost = lastTime.until(metaData.tstamp, ChronoUnit.MILLIS);
            }
            result.add(new Pair<>(lastInstruction, cost));
            lastInstruction = currInstruction;
            lastTime = metaData.tstamp;
            lastNumDeletions = metaData.deletionsUpToNow;
        }

        var successTimeR = jooq.select(GAME_LOGS.TIMESTAMP)
                .from(GAME_LOGS)
                .where(GAME_LOGS.GAMEID.eq(gameId))
                .and(GAME_LOGS.MESSAGE.contains("SuccessfullyFinished"))
                .fetchOne();
        
        if (successTimeR != null && lastTime != null) {
            if (costIsNumErrors) {
                result.add(new Pair<>(lastInstruction, (long)(deletionsUpToNow - lastNumDeletions)));
            } else {
                result.add(new Pair<>(lastInstruction, lastTime.until(successTimeR.component1(), ChronoUnit.MILLIS)));
            }
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
    
    public void evaluateModelFit() {
        List<Pair<Long, Long>> result = new ArrayList<>();
        var model = predictDurationCoeffsFromAllGames();
        for (var game: allData) {
            boolean start_of_game = true;
            for (var instruction: game) {
                var features = instruction.left;
                var duration = instruction.right;
                long prediction = 0;
                if (start_of_game) {
                    prediction += model.weights.get(FIRST_INSTRUCTION_FEATURE);
                    start_of_game = false;
                }
                for (var f: features) {
                    if (f.startsWith(WeightEstimator.FIRST_OCCURENCE_PREFIX)) {
                        prediction += model.firstOccurenceWeights.get(f.substring(FIRST_OCCURENCE_PREFIX.length()));
                    } else {
                        prediction += model.weights.get(f);
                    }
                }
                result.add(new Pair<>(duration, prediction));
            }
        }

        double sample_average = result.stream().collect(Collectors.averagingLong(x->x.left));
        double predction_average = result.stream().collect(Collectors.averagingLong(x->x.right));
        double sample_variance = result.stream().collect(Collectors.averagingDouble(
                (x) -> Math.pow(x.left - sample_average, 2)));
        double mean_squared_residuals = result.stream().collect(Collectors.averagingDouble(
                (x) -> Math.pow(x.left - x.right, 2)));
        
        System.out.println("Average actual duration: " + sample_average);
        System.out.println("Average predicted duration: " + predction_average);
        System.out.println("Average absolute error: " +
                result.stream().collect(Collectors.averagingLong(x-> Math.abs(x.left - x.right))));
        System.out.println("Sample variance: " + sample_variance);
        System.out.println("Mean squared residuals: " + mean_squared_residuals);
        System.out.println("R squared: " +  (1 - (mean_squared_residuals / sample_variance)));
    }
    
}
