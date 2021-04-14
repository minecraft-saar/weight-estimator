package de.saar.minecraft.analysis;

import picocli.CommandLine;

import java.util.ArrayList;
import java.util.concurrent.Callable;

@CommandLine.Command(name = "WeightEstimator", mixinStandardHelpOptions = true,
        description = "Estimates grammar weights from played games.")
public class WeightEstimatorCLI implements Callable<Integer> {
    enum EstimationMode {OPTIMAL, BOOTSTRAPPED, BOTH}
    @CommandLine.Parameters(index = "0", description = "Connection String to connect to the DB")
    String connStr;

    @CommandLine.Option(names = "--db-user")
    String dbUser = "minecraft";

    @CommandLine.Option(names = "--db-pass")
    String dbPass = "";

    @CommandLine.Option(names = {"-l", "--lower-percentile"})
    int lowerPercentile = 10;

    @CommandLine.Option(names = {"-u", "--upper-percentile"})
    int upperPercentile = 90;

    @CommandLine.Option(names = "--mode",
            description = "OPTIMAL (default): LR on all data; BOOTSTRAPPED: sample from bootstrapped LRs, BOTH: both")
    EstimationMode mode = EstimationMode.BOTH;

    @CommandLine.Option(names = "--architect",
            description = "Restrict to architect with this name. Leave empty for no restriction.")
    String architectName = "";
    
    @CommandLine.Option(names = "--sample-individual-instructions",
            description = "Bootstrap sampling on the instruction level instead of the game level")
    boolean sampleIndividualInstructions;

    @Override
    public Integer call(){

        var estimator = new WeightEstimator(connStr, dbUser, dbPass,
                lowerPercentile,upperPercentile,
                new ArrayList<>(), // seed games not used in the CLI
                architectName );
        
        if (mode == EstimationMode.OPTIMAL || mode == EstimationMode.BOTH) {
            var results = estimator.predictDurationCoeffsFromAllGames();
            System.out.println("global optimum:");
            System.out.println(results.toJson());
        }

        if (mode == EstimationMode.BOOTSTRAPPED || mode == EstimationMode.BOTH) {
            System.out.println("bootstrapped:");
            var results = estimator.sampleDurationCoeffsWithBootstrap(1000,
                    !sampleIndividualInstructions);
            System.out.println(results.toJson());
        }
        return 0;
    }
}
