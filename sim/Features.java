import mage.abilities.keyword.FlyingAbility;
import mage.counters.CounterType;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.permanent.PermanentToken;
import mage.player.ai.score.GameStateEvaluator2;
import mage.players.Player;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * A position as a fixed list of numbers, from one player's point of view.
 *
 * Shared by the data logger (SimData) and the learned evaluator, so the numbers
 * a model is trained on are computed by exactly the code that later feeds it --
 * no Python/Java re-implementation to drift apart.
 */
public final class Features {

    static final String[] SIDE = {"life", "hand", "library", "graveyard", "lands", "lands_untapped",
            "creatures", "power", "toughness", "power_untapped", "power_flying", "power_max",
            "nonland_mv", "other_permanents", "loyalty", "tokens"};

    public static List<String> names() {
        List<String> out = new ArrayList<>();
        for (String s : SIDE) out.add("me_" + s);
        for (String s : SIDE) out.add("opp_" + s);
        out.add("turn");
        out.add("my_turn");
        out.add("stack");
        out.add("xmage_score");
        return out;
    }

    public static double[] of(Game game, UUID me) {
        Player p = game.getPlayer(me);
        UUID oppId = game.getOpponents(me).iterator().next();
        Player o = game.getPlayer(oppId);
        double[] f = new double[SIDE.length * 2 + 4];
        side(game, p, f, 0);
        side(game, o, f, SIDE.length);
        int k = SIDE.length * 2;
        f[k] = game.getTurnNum();
        f[k + 1] = me.equals(game.getActivePlayerId()) ? 1 : 0;
        f[k + 2] = game.getStack().size();
        // XMage's own (stock) evaluation -- never the learned override, which feeds on this --
        // clipped away from its +-1e8 win/lose sentinels
        double xs = GameStateEvaluator2.evaluateStock(me, game, true).getTotalScore();
        f[k + 3] = Math.max(-5000, Math.min(5000, xs));
        return f;
    }

    private static void side(Game game, Player p, double[] f, int at) {
        f[at] = p.getLife();
        f[at + 1] = p.getHand().size();
        f[at + 2] = p.getLibrary().size();
        f[at + 3] = p.getGraveyard().size();
        int lands = 0, landsU = 0, creatures = 0, power = 0, tough = 0, powerU = 0, powerF = 0,
                powerMax = 0, mv = 0, other = 0, loyalty = 0, tokens = 0;
        for (Permanent perm : game.getBattlefield().getAllActivePermanents(p.getId())) {
            if (perm instanceof PermanentToken) tokens++;
            if (perm.isLand(game)) {
                lands++;
                if (!perm.isTapped()) landsU++;
                if (!perm.isCreature(game)) continue;
            }
            mv += perm.getManaValue();
            if (perm.isCreature(game)) {
                int pw = Math.max(0, perm.getPower().getValue());
                creatures++;
                power += pw;
                tough += Math.max(0, perm.getToughness().getValue());
                if (!perm.isTapped()) powerU += pw;
                if (perm.getAbilities(game).containsKey(FlyingAbility.getInstance().getId())) powerF += pw;
                powerMax = Math.max(powerMax, pw);
            } else {
                other++;
            }
            if (perm.isPlaneswalker(game)) loyalty += perm.getCounters(game).getCount(CounterType.LOYALTY);
        }
        f[at + 4] = lands;
        f[at + 5] = landsU;
        f[at + 6] = creatures;
        f[at + 7] = power;
        f[at + 8] = tough;
        f[at + 9] = powerU;
        f[at + 10] = powerF;
        f[at + 11] = powerMax;
        f[at + 12] = mv;
        f[at + 13] = other;
        f[at + 14] = loyalty;
        f[at + 15] = tokens;
    }
}
