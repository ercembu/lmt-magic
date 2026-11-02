import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import mage.game.Game;
import mage.player.ai.score.GameStateEvaluator2;
import mage.players.Player;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;

/**
 * A learned position evaluator (trained by scripts/eval_train.py) plugged into
 * XMage's look-ahead through the GameStateEvaluator2.OVERRIDE hook.
 *
 * It only answers for players whose name ends in "+", so a head-to-head can pit
 * a learned AI against the stock one inside the same JVM; everyone else falls
 * through to XMage's hand-written score. The AI simulates by copying players,
 * and the copies keep their names, so the "+" survives into its look-ahead.
 */
public final class LearnedEval implements GameStateEvaluator2.Override {

    private final double[] mean, scale;
    private final double[][][] W;      // [layer][out][in]
    private final double[][] b;
    private final boolean relu;
    private final double gain;

    public LearnedEval(Path modelJson, double gain) throws Exception {
        JsonObject m = JsonParser.parseString(Files.readString(modelJson)).getAsJsonObject();
        List<String> want = Features.names();
        JsonArray feats = m.getAsJsonArray("features");
        if (feats.size() != want.size()) {
            throw new IllegalStateException("model has " + feats.size() + " features, Features has " + want.size());
        }
        for (int i = 0; i < want.size(); i++) {
            if (!feats.get(i).getAsString().equals(want.get(i))) {
                throw new IllegalStateException("feature " + i + " is " + feats.get(i) + ", expected " + want.get(i));
            }
        }
        mean = vec(m.getAsJsonArray("mean"));
        scale = vec(m.getAsJsonArray("scale"));
        JsonArray layers = m.getAsJsonArray("layers");
        W = new double[layers.size()][][];
        b = new double[layers.size()][];
        for (int l = 0; l < layers.size(); l++) {
            JsonObject L = layers.get(l).getAsJsonObject();
            JsonArray rows = L.getAsJsonArray("W");
            W[l] = new double[rows.size()][];
            for (int r = 0; r < rows.size(); r++) W[l][r] = vec(rows.get(r).getAsJsonArray());
            b[l] = vec(L.getAsJsonArray("b"));
        }
        relu = "relu".equals(m.get("hidden_activation").getAsString());
        this.gain = gain;
    }

    private static double[] vec(JsonArray a) {
        double[] v = new double[a.size()];
        for (int i = 0; i < v.length; i++) v[i] = a.get(i).getAsDouble();
        return v;
    }

    /** P(this player wins) from the position. */
    public double winProbability(Game game, UUID playerId) {
        double[] x = Features.of(game, playerId);
        for (int i = 0; i < x.length; i++) x[i] = (x[i] - mean[i]) / (scale[i] == 0 ? 1 : scale[i]);
        for (int l = 0; l < W.length; l++) {
            double[] y = new double[W[l].length];
            for (int o = 0; o < y.length; o++) {
                double s = b[l][o];
                for (int i = 0; i < x.length; i++) s += W[l][o][i] * x[i];
                y[o] = (relu && l < W.length - 1) ? Math.max(0, s) : s;
            }
            x = y;
        }
        return 1 / (1 + Math.exp(-x[0]));
    }

    @Override
    public Integer score(UUID playerId, Game game) {
        Player p = game.getPlayer(playerId);
        if (p == null || !p.getName().endsWith("+")) return null;
        double pr = Math.min(1 - 1e-6, Math.max(1e-6, winProbability(game, playerId)));
        // log-odds, scaled into the integer range XMage's search compares
        return (int) Math.round(gain * Math.log(pr / (1 - pr)));
    }
}
