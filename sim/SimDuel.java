import mage.cards.decks.Deck;
import mage.cards.decks.DeckCardLists;
import mage.cards.decks.importer.DeckImporter;
import mage.cards.repository.CardScanner;
import mage.constants.MultiplayerAttackOption;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameOptions;
import mage.game.TwoPlayerDuel;
import mage.game.TwoPlayerMatch;
import mage.game.match.Match;
import mage.game.match.MatchOptions;
import mage.game.mulligan.MulliganType;
import mage.player.ai.score.GameStateEvaluator2;
import mage.players.Player;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Learned AI vs stock AI. Each deck pairing is played twice with the learned AI
 * swapping decks, so deck strength cancels and only the evaluator differs.
 *
 *   java -cp patch:... SimDuel pairs.tsv model.json results.tsv [threads] [gain] [skill] [maxTurns]
 *
 * results.tsv: pairIndex  learnedDeck  stockDeck  winner(learned|stock|draw|error)  turns
 * The patched GameStateEvaluator2 (sim/patch) must come first on the classpath.
 */
public class SimDuel {

    public static void main(String[] args) throws Exception {
        List<String[]> pairs = new ArrayList<>();
        for (String l : Files.readAllLines(Paths.get(args[0]))) {
            String[] p = l.split("\t");
            if (p.length == 2) pairs.add(p);
        }
        Path model = Paths.get(args[1]);
        Path outFile = Paths.get(args[2]);
        int threads = args.length > 3 ? Integer.parseInt(args[3]) : 8;
        double gain = args.length > 4 ? Double.parseDouble(args[4]) : 200;
        int skill = args.length > 5 ? Integer.parseInt(args[5]) : 1;
        int maxTurns = args.length > 6 ? Integer.parseInt(args[6]) : 30;

        CardScanner.scan();
        mage.cards.repository.TokenRepository.instance.getAll();
        GameStateEvaluator2.OVERRIDE = new LearnedEval(model, gain);

        PrintWriter out = new PrintWriter(new FileWriter(outFile.toFile(), true));
        AtomicInteger tid = new AtomicInteger(), n = new AtomicInteger(), learnedWins = new AtomicInteger(),
                decided = new AtomicInteger();
        ExecutorService pool = Executors.newFixedThreadPool(threads,
                r -> new Thread(r, "GAME duel-" + tid.incrementAndGet()));
        for (int i = 0; i < pairs.size(); i++) {
            for (int swap = 0; swap < 2; swap++) {
                final int idx = i;
                final String learnedDeck = pairs.get(i)[swap], stockDeck = pairs.get(i)[1 - swap];
                pool.submit(() -> {
                    String res;
                    try {
                        res = play(learnedDeck, stockDeck, skill, maxTurns);
                    } catch (Throwable e) {
                        res = "error\t0";
                        System.err.println("error: " + e);
                    }
                    if (res.startsWith("learned") || res.startsWith("stock")) {
                        decided.incrementAndGet();
                        if (res.startsWith("learned")) learnedWins.incrementAndGet();
                    }
                    synchronized (out) {
                        out.println(idx + "\t" + Paths.get(learnedDeck).getFileName() + "\t"
                                + Paths.get(stockDeck).getFileName() + "\t" + res);
                        out.flush();
                    }
                    int k = n.incrementAndGet();
                    if (k % 20 == 0) {
                        int d = Math.max(1, decided.get());
                        System.err.printf("%d/%d games, learned AI won %d/%d = %.1f%%%n",
                                k, 2 * pairs.size(), learnedWins.get(), d, 100.0 * learnedWins.get() / d);
                    }
                });
            }
        }
        pool.shutdown();
        pool.awaitTermination(30, TimeUnit.DAYS);
        out.close();
        System.exit(0);
    }

    static String play(String learnedDeck, String stockDeck, int skill, int maxTurns) throws Exception {
        Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ALL,
                MulliganType.GAME_DEFAULT.getMulligan(0), 40, 20, 7);
        Match match = new TwoPlayerMatch(new MatchOptions("sim", "Two Player Duel", false));
        Player l = add(game, match, "L+", learnedDeck, skill);
        Player s = add(game, match, "S", stockDeck, skill);
        GameOptions opts = new GameOptions();
        opts.stopOnTurn = maxTurns;
        game.setGameOptions(opts);
        game.start(new Random().nextBoolean() ? l.getId() : s.getId());
        String w = game.getWinner();
        String res = w.contains("Player L+ ") ? "learned" : w.contains("Player S ") ? "stock" : "draw";
        return res + "\t" + game.getTurnNum();
    }

    static Player add(Game game, Match match, String name, String deckPath, int skill) throws Exception {
        DeckCardLists list = DeckImporter.importDeckFromFile(deckPath, false);
        Deck deck = Deck.load(list, false, false);
        Player p = new SimAI(name, skill);
        game.loadCards(deck.getCards(), p.getId());
        game.addPlayer(p, deck);
        match.addPlayer(p, deck);
        return p;
    }
}
