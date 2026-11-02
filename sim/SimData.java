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
import mage.players.Player;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPOutputStream;

/**
 * Training data for a learned position evaluator: plays games and, at every new
 * step of the real game, writes one row per player -- Features.of() from that
 * player's side -- then labels every row with whether that player won.
 *
 *   java -cp ... SimData pairs.tsv out.tsv.gz [threads] [skill] [maxTurns] [firstGameId]
 *
 * Rows of one game are written together, so a train/test split can hold out
 * whole games (positions from the same game are not independent).
 */
public class SimData {

    static class Log {
        final Game real;
        final List<String> rows = new ArrayList<>();
        String last = "";

        Log(Game real) {
            this.real = real;
        }

        synchronized void snap(Game game) {
            if (game != real) return;
            String key = game.getTurnNum() + "/" + game.getTurnStepType();
            if (key.equals(last)) return;
            last = key;
            for (UUID id : game.getPlayers().keySet()) {
                Player p = game.getPlayer(id);
                if (p == null || p.hasLost() || p.hasWon()) continue;
                double[] f = Features.of(game, id);
                StringBuilder sb = new StringBuilder(p.getName()).append('\t')
                        .append(game.getTurnNum()).append('\t').append(game.getTurnStepType());
                for (double v : f) sb.append('\t').append(v == Math.rint(v) ? String.valueOf((long) v) : String.valueOf(v));
                rows.add(sb.toString());
            }
        }
    }

    static class DataAI extends SimAI {
        Log log;

        DataAI(String name, int skill) {
            super(name, skill);
        }

        DataAI(final DataAI p) {
            super(p);
            this.log = p.log;
        }

        @Override
        public DataAI copy() {
            return new DataAI(this);
        }

        @Override
        public boolean priority(Game game) {
            if (log != null) log.snap(game);
            return super.priority(game);
        }
    }

    public static void main(String[] args) throws Exception {
        Path pairsFile = Paths.get(args[0]);
        Path outFile = Paths.get(args[1]);
        int threads = args.length > 2 ? Integer.parseInt(args[2]) : 4;
        int skill = args.length > 3 ? Integer.parseInt(args[3]) : 1;
        int maxTurns = args.length > 4 ? Integer.parseInt(args[4]) : 30;
        int firstId = args.length > 5 ? Integer.parseInt(args[5]) : 0;

        CardScanner.scan();
        mage.cards.repository.TokenRepository.instance.getAll();

        List<String[]> todo = new ArrayList<>();
        for (String l : Files.readAllLines(pairsFile)) {
            String[] p = l.split("\t");
            if (p.length == 2) todo.add(p);
        }
        Writer out = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(outFile), true),
                StandardCharsets.UTF_8);
        out.write("game\tdeckA\tdeckB\tseat\tturn\tstep\t" + String.join("\t", Features.names()) + "\twon\n");
        AtomicInteger tid = new AtomicInteger(), n = new AtomicInteger();
        ExecutorService pool = Executors.newFixedThreadPool(threads,
                r -> new Thread(r, "GAME data-" + tid.incrementAndGet()));
        for (int i = 0; i < todo.size(); i++) {
            final int gid = firstId + i;
            final String[] pair = todo.get(i);
            pool.submit(() -> {
                try {
                    play(gid, pair[0], pair[1], skill, maxTurns, out);
                } catch (Throwable e) {
                    System.err.println("error " + pair[0] + " vs " + pair[1] + ": " + e);
                }
                int k = n.incrementAndGet();
                if (k % 20 == 0) System.err.println(k + "/" + todo.size() + " games logged");
            });
        }
        pool.shutdown();
        pool.awaitTermination(30, TimeUnit.DAYS);
        out.close();
        System.exit(0);
    }

    static void play(int gid, String a, String b, int skill, int maxTurns, Writer out) throws Exception {
        Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ALL,
                MulliganType.GAME_DEFAULT.getMulligan(0), 40, 20, 7);
        Match match = new TwoPlayerMatch(new MatchOptions("sim", "Two Player Duel", false));
        Log log = new Log(game);
        DataAI pa = add(game, match, "A", a, skill, log);
        DataAI pb = add(game, match, "B", b, skill, log);
        GameOptions opts = new GameOptions();
        opts.stopOnTurn = maxTurns;
        game.setGameOptions(opts);
        game.start(new Random().nextBoolean() ? pa.getId() : pb.getId());
        String w = game.getWinner();
        String winner = w.contains("Player A ") ? "A" : w.contains("Player B ") ? "B" : "";
        String da = Paths.get(a).getFileName().toString(), db = Paths.get(b).getFileName().toString();
        StringBuilder sb = new StringBuilder();
        for (String row : log.rows) {
            String seat = row.substring(0, row.indexOf('\t'));
            String won = winner.isEmpty() ? "0.5" : seat.equals(winner) ? "1" : "0";
            sb.append(gid).append('\t').append(da).append('\t').append(db).append('\t')
                    .append(row).append('\t').append(won).append('\n');
        }
        synchronized (out) {
            out.write(sb.toString());
            out.flush();
        }
    }

    static DataAI add(Game game, Match match, String name, String deckPath, int skill, Log log)
            throws Exception {
        DeckCardLists list = DeckImporter.importDeckFromFile(deckPath, false);
        Deck deck = Deck.load(list, false, false);
        DataAI p = new DataAI(name, skill);
        p.log = log;
        game.loadCards(deck.getCards(), p.getId());
        game.addPlayer(p, deck);
        match.addPlayer(p, deck);
        return p;
    }
}
