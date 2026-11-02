import mage.abilities.Ability;
import mage.cards.Card;
import mage.constants.Outcome;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.player.ai.ComputerPlayer7;
import mage.target.Target;
import mage.target.common.TargetCardInHand;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * XMage's ComputerPlayer7 with one decision fixed: which cards go to the bottom
 * after a London mulligan.
 *
 * XMage asks the AI with its *discard* logic, which happily throws away lands. In
 * recorded games that showed up as players mulliganing twice, drawing a fine
 * seven, then bottoming two lands and keeping five spells and no land at all.
 * Here the bottomed cards steer the kept hand toward the deck's land ratio
 * (17 of 40): bottom a land if the hand is land-heavy, else the most expensive
 * spell. Every other decision is the stock AI.
 */
public class SimAI extends ComputerPlayer7 {

    private static final Pattern MORE = Pattern.compile("\\((\\d+) more\\)");
    private static final double LAND_SHARE = 17.0 / 40.0;

    public SimAI(String name, int skill) {
        super(name, RangeOfInfluence.ALL, skill);
    }

    public SimAI(final SimAI p) {
        super(p);
    }

    @Override
    public SimAI copy() {
        return new SimAI(this);
    }

    @Override
    public boolean chooseTarget(Outcome outcome, Target target, Ability source, Game game) {
        if (source == null && target instanceof TargetCardInHand) {
            String msg = target.getFilter().getMessage();
            Matcher m = MORE.matcher(msg);
            if (msg.contains("bottom of your library") && m.find()) {
                int toBottom = Integer.parseInt(m.group(1));
                List<Card> hand = new ArrayList<>(getHand().getCards(game));
                int keep = hand.size() - toBottom;
                long lands = hand.stream().filter(c -> c.isLand(game)).count();
                int want = Math.max(2, Math.min(keep - 1, (int) Math.round(keep * LAND_SHARE)));
                Card pick;
                if (lands > want) {
                    pick = hand.stream().filter(c -> c.isLand(game)).findFirst().orElse(null);
                } else {
                    pick = hand.stream().filter(c -> !c.isLand(game))
                            .max(Comparator.comparingInt(Card::getManaValue)).orElse(null);
                }
                if (pick != null) {
                    target.addTarget(pick.getId(), null, game);
                    return true;
                }
            }
        }
        return super.chooseTarget(outcome, target, source, game);
    }
}
