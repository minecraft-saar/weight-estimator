package de.saar.minecraft.analysis;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

public class WeightEstimatorSerializationTest {
    @Test
    public void serializationTest() {
        HashMap<String, Double> weights = new HashMap<>();
        // we have one set of weights ...
        weights.put("block", 1.123);
        // but this one will end up in the firstOccurenceWeights:
        weights.put(WeightEstimator.FIRST_OCCURENCE_PREFIX + "block", 1.123);
        WeightEstimator.WeightResult wr = new WeightEstimator.WeightResult(weights);
        String json = wr.toJson();
        WeightEstimator.WeightResult wr2 = WeightEstimator.WeightResult.fromJson(json);
        assertEquals(1, wr2.weights.size() );
        assertEquals(1, wr2.firstOccurenceWeights.size() );
    }
}
