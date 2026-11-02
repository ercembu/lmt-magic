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
import mage.cards.decks.DeckCardInfo;
import mage.cards.repository.CardRepository;
import mage.game.mulligan.MulliganType;
import mage.player.ai.ComputerPlayer7;
import mage.players.Player;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Plays AI-vs-AI limited games between pairs of decks and writes one line per game.
 *
 *   java -cp ... SimRunner pairs.tsv results.tsv [threads] [skill] [maxTurns]
 *
 * pairs.tsv: "deckA.dck<TAB>deckB.dck" per line. results.tsv gets
 * "deckA deckB winner(A|B|draw|error) turns seconds", appended and flushed per game,
 * so a run can be stopped and resumed (finished pairs are skipped).
 */
public class SimRunner {

    public static void main(String[] args) throws Exception {
        Path pairsFile = Paths.get(args[0]);
        Path outFile = Paths.get(args[1]);
        int threads = args.length > 2 ? Integer.parseInt(args[2]) : 8;
        int skill = args.length > 3 ? Integer.parseInt(args[3]) : 1;
        int maxTurns = args.length > 4 ? Integer.parseInt(args[4]) : 30;

        CardScanner.scan();
        // the token repository loads lazily and is not thread-safe: with 24 games
        // starting at once, ~30% of the first batch died in it before a card was
        // played. Load it once, up front.
        mage.cards.repository.TokenRepository.instance.getAll();

        Set<String> done = new HashSet<>();
        if (Files.exists(outFile)) {
            for (String l : Files.readAllLines(outFile)) {
                String[] p = l.split("\t");
                // errored games are retried
                if (p.length >= 3 && !p[2].equals("error")) done.add(p[0] + "\t" + p[1]);
            }
        }
        List<String[]> todo = new ArrayList<>();
        for (String l : Files.readAllLines(pairsFile)) {
            String[] p = l.split("\t");
            if (p.length == 2 && !done.contains(p[0] + "\t" + p[1])) todo.add(p);
        }
        System.err.println(todo.size() + " games to play (" + done.size() + " already done), "
                + threads + " threads, skill " + skill);

        Map<String, DeckCardLists> deckCache = new ConcurrentHashMap<>();
        PrintWriter out = new PrintWriter(new FileWriter(outFile.toFile(), true));
        // XMage only lets game code run on threads whose name starts with "GAME"
        AtomicInteger tid = new AtomicInteger();
        ExecutorService pool = Executors.newFixedThreadPool(threads,
                r -> new Thread(r, "GAME sim-" + tid.incrementAndGet()));
        AtomicInteger n = new AtomicInteger();
        long t0 = System.currentTimeMillis();
        for (String[] pair : todo) {
            pool.submit(() -> {
                String line;
                long g0 = System.currentTimeMillis();
                try {
                    line = play(pair[0], pair[1], deckCache, skill, maxTurns);
                } catch (Throwable e) {
                    line = "error\t0";
                    StringBuilder st = new StringBuilder();
                    for (StackTraceElement el : e.getStackTrace()) {
                        if (st.length() > 1500) break;
                        st.append("\n    at ").append(el);
                    }
                    System.err.println("error " + pair[0] + " vs " + pair[1] + ": " + e + st);
                }
                double secs = (System.currentTimeMillis() - g0) / 1000.0;
                synchronized (out) {
                    out.println(pair[0] + "\t" + pair[1] + "\t" + line + "\t" + String.format("%.1f", secs));
                    out.flush();
                }
                int k = n.incrementAndGet();
                if (k % 10 == 0 || k == todo.size()) {
                    double el = (System.currentTimeMillis() - t0) / 1000.0;
                    System.err.printf("%d/%d games, %.1f games/min%n", k, todo.size(), 60.0 * k / el);
                }
            });
        }
        pool.shutdown();
        pool.awaitTermination(30, TimeUnit.DAYS);
        out.close();
        System.exit(0);
    }

    static String play(String a, String b, Map<String, DeckCardLists> cache, int skill, int maxTurns)
            throws Exception {
        Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ALL,
                MulliganType.GAME_DEFAULT.getMulligan(0), 40, 20, 7);
        // the AI copies its match record when it simulates ahead, so it needs one
        Match match = new TwoPlayerMatch(new MatchOptions("sim", "Two Player Duel", false));
        Player pa = addPlayer(game, match, "A", a, cache, skill);
        Player pb = addPlayer(game, match, "B", b, cache, skill);
        GameOptions opts = new GameOptions();
        opts.stopOnTurn = maxTurns;
        game.setGameOptions(opts);
        // who chooses play/draw alternates with the coin flip the engine does itself
        game.start(new Random().nextBoolean() ? pa.getId() : pb.getId());
        String w = game.getWinner();
        String res = w.contains("Player A ") ? "A" : w.contains("Player B ") ? "B" : "draw";
        return res + "\t" + game.getTurnNum();
    }

    static Player addPlayer(Game game, Match match, String name, String deckPath,
                            Map<String, DeckCardLists> cache, int skill) throws Exception {
        // parsed per game: XMage's deck loader mutates the list, so sharing one
        // between threads threw ConcurrentModificationException in ~1% of games
        DeckCardLists list = DeckImporter.importDeckFromFile(deckPath, false);
        Deck deck = Deck.load(list, false, false);
        if (deck.getMaindeckCards().size() < 40) {
            StringBuilder miss = new StringBuilder();
            for (DeckCardInfo c : list.getCards()) {
                if (CardRepository.instance.findCard(c.getSetCode(), c.getCardNumber()) == null) {
                    miss.append(" [").append(c.getSetCode()).append(':').append(c.getCardNumber())
                            .append("] ").append(c.getCardName()).append(';');
                }
            }
            Map<String, Integer> want = new TreeMap<>(), got = new TreeMap<>();
            for (DeckCardInfo c : list.getCards()) want.merge(c.getCardName(), c.getAmount(), Integer::sum);
            for (mage.cards.Card c : deck.getMaindeckCards()) got.merge(c.getName(), 1, Integer::sum);
            for (String k : want.keySet()) {
                if (!want.get(k).equals(got.getOrDefault(k, 0))) {
                    miss.append(" ").append(k).append(" want=").append(want.get(k))
                            .append(" got=").append(got.getOrDefault(k, 0)).append(';');
                }
            }
            throw new IllegalStateException("deck " + deckPath + " has " + deck.getMaindeckCards().size()
                    + "; mismatched:" + miss + " loaded-names=" + got.keySet());
        }
        Player p = new SimAI(name, skill);
        game.loadCards(deck.getCards(), p.getId());
        game.addPlayer(p, deck);
        match.addPlayer(p, deck);
        return p;
    }
}
