import mage.cards.Card;
import mage.cards.decks.Deck;
import mage.cards.decks.DeckCardLists;
import mage.cards.decks.importer.DeckImporter;
import mage.cards.repository.CardScanner;
import mage.constants.MultiplayerAttackOption;
import mage.constants.RangeOfInfluence;
import mage.counters.Counter;
import mage.game.Game;
import mage.game.GameOptions;
import mage.game.TwoPlayerDuel;
import mage.game.TwoPlayerMatch;
import mage.game.events.TableEvent;
import mage.game.match.Match;
import mage.game.match.MatchOptions;
import mage.game.mulligan.MulliganType;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.player.ai.ComputerPlayer7;
import mage.players.Player;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPOutputStream;

/**
 * Plays games like SimRunner, but records each one for the viewer: a snapshot of
 * both boards every time the real game hands out priority, plus the game's own
 * log lines in between. Written as one gzipped JSON file per game.
 *
 *   java -cp ... SimRecorder pairs.tsv outdir [threads] [skill] [maxTurns]
 *
 * The AI plans by copying the game and playing out hypothetical futures, so a
 * snapshot is only taken when the game is the real one -- otherwise the replay
 * would be full of moves nobody made.
 */
public class SimRecorder {

    static class Recording {
        final Game real;
        final List<String> snaps = new ArrayList<>();
        final List<String> pendingLog = new ArrayList<>();
        String lastState = "";

        Recording(Game real) {
            this.real = real;
        }

        synchronized void log(String msg) {
            pendingLog.add(msg);
        }

        synchronized void snap(Game game, boolean force) {
            if (game != real) return;
            String state = state(game);
            if (!force && state.equals(lastState) && pendingLog.isEmpty()) return;
            lastState = state;
            StringBuilder sb = new StringBuilder("{\"turn\":").append(game.getTurnNum())
                    .append(",\"step\":").append(q(String.valueOf(game.getTurnStepType())))
                    .append(",\"active\":").append(q(name(game, game.getActivePlayerId())))
                    .append(",\"log\":[");
            for (int i = 0; i < pendingLog.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(q(pendingLog.get(i)));
            }
            pendingLog.clear();
            sb.append("],").append(state).append('}');
            snaps.add(sb.toString());
        }
    }

    /** SimAI (stock AI, fixed mulligan bottoming) that snapshots the board whenever the real game gives it priority. */
    static class RecordingAI extends SimAI {
        Recording rec;

        RecordingAI(String name, int skill) {
            super(name, skill);
        }

        RecordingAI(final RecordingAI p) {
            super(p);
            this.rec = p.rec;
        }

        @Override
        public RecordingAI copy() {
            return new RecordingAI(this);
        }

        @Override
        public boolean priority(Game game) {
            if (rec != null) rec.snap(game, false);
            return super.priority(game);
        }
    }

    static String name(Game game, UUID id) {
        Player p = id == null ? null : game.getPlayer(id);
        return p == null ? "" : p.getName();
    }

    static String q(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (char c : (s == null ? "" : s).toCharArray()) {
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': break;
                case '\t': sb.append(' '); break;
                default:
                    if (c < 0x20) sb.append(' ');
                    else sb.append(c);
            }
        }
        return sb.append('"').toString();
    }

    static String cards(Collection<Card> cs) {
        StringBuilder sb = new StringBuilder("[");
        boolean first = true;
        for (Card c : cs) {
            if (!first) sb.append(',');
            first = false;
            sb.append(q(c.getName()));
        }
        return sb.append(']').toString();
    }

    /** Both players' visible state, as the "players" and "stack" JSON fields. */
    static String state(Game game) {
        StringBuilder sb = new StringBuilder("\"players\":[");
        boolean firstP = true;
        for (Player p : game.getPlayers().values()) {
            if (!firstP) sb.append(',');
            firstP = false;
            sb.append("{\"name\":").append(q(p.getName()))
                    .append(",\"life\":").append(p.getLife())
                    .append(",\"library\":").append(p.getLibrary().size())
                    .append(",\"hand\":").append(cards(p.getHand().getCards(game)))
                    .append(",\"graveyard\":").append(cards(p.getGraveyard().getCards(game)))
                    .append(",\"battlefield\":[");
            boolean firstC = true;
            for (Permanent perm : game.getBattlefield().getAllActivePermanents(p.getId())) {
                if (!firstC) sb.append(',');
                firstC = false;
                sb.append("{\"name\":").append(q(perm.getName()))
                        .append(",\"tapped\":").append(perm.isTapped())
                        .append(",\"token\":").append(perm instanceof mage.game.permanent.PermanentToken)
                        .append(",\"land\":").append(perm.isLand(game))
                        .append(",\"creature\":").append(perm.isCreature(game));
                if (perm.isCreature(game)) {
                    sb.append(",\"pt\":").append(q(perm.getPower().getValue() + "/" + perm.getToughness().getValue()));
                }
                if (perm.isAttacking()) sb.append(",\"attacking\":true");
                if (perm.getBlocking() > 0) sb.append(",\"blocking\":true");
                if (perm.getDamage() > 0) sb.append(",\"damage\":").append(perm.getDamage());
                StringBuilder ctr = new StringBuilder();
                for (Counter c : perm.getCounters(game).values()) {
                    if (ctr.length() > 0) ctr.append(", ");
                    ctr.append(c.getCount()).append(' ').append(c.getName());
                }
                if (ctr.length() > 0) sb.append(",\"counters\":").append(q(ctr.toString()));
                sb.append('}');
            }
            sb.append("]}");
        }
        sb.append("],\"stack\":[");
        boolean firstS = true;
        for (StackObject so : game.getStack()) {
            if (!firstS) sb.append(',');
            firstS = false;
            sb.append(q(so.getName() + " (" + name(game, so.getControllerId()) + ")"));
        }
        return sb.append(']').toString();
    }

    public static void main(String[] args) throws Exception {
        Path pairsFile = Paths.get(args[0]);
        Path outDir = Paths.get(args[1]);
        int threads = args.length > 2 ? Integer.parseInt(args[2]) : 8;
        int skill = args.length > 3 ? Integer.parseInt(args[3]) : 1;
        int maxTurns = args.length > 4 ? Integer.parseInt(args[4]) : 30;
        Files.createDirectories(outDir);

        CardScanner.scan();
        mage.cards.repository.TokenRepository.instance.getAll();

        List<String[]> todo = new ArrayList<>();
        for (String l : Files.readAllLines(pairsFile)) {
            String[] p = l.split("\t");
            if (p.length == 2) todo.add(p);
        }
        AtomicInteger tid = new AtomicInteger(), n = new AtomicInteger();
        ExecutorService pool = Executors.newFixedThreadPool(threads,
                r -> new Thread(r, "GAME rec-" + tid.incrementAndGet()));
        for (int i = 0; i < todo.size(); i++) {
            final int idx = i;
            final String[] pair = todo.get(i);
            pool.submit(() -> {
                Path out = outDir.resolve(String.format("G%05d.json.gz", idx));
                if (Files.exists(out)) return;
                try {
                    String json = play(pair[0], pair[1], skill, maxTurns);
                    try (Writer w = new OutputStreamWriter(
                            new GZIPOutputStream(Files.newOutputStream(out)), StandardCharsets.UTF_8)) {
                        w.write(json);
                    }
                } catch (Throwable e) {
                    System.err.println("error " + pair[0] + " vs " + pair[1] + ": " + e);
                }
                int k = n.incrementAndGet();
                if (k % 10 == 0) System.err.println(k + "/" + todo.size() + " recorded");
            });
        }
        pool.shutdown();
        pool.awaitTermination(30, TimeUnit.DAYS);
        System.exit(0);
    }

    static String play(String a, String b, int skill, int maxTurns) throws Exception {
        Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ALL,
                MulliganType.GAME_DEFAULT.getMulligan(0), 40, 20, 7);
        Match match = new TwoPlayerMatch(new MatchOptions("sim", "Two Player Duel", false));
        Recording rec = new Recording(game);
        RecordingAI pa = addPlayer(game, match, "A", a, skill, rec);
        RecordingAI pb = addPlayer(game, match, "B", b, skill, rec);
        game.addTableEventListener(e -> {
            if (e.getEventType() == TableEvent.EventType.INFO) rec.log(e.getMessage());
        });
        GameOptions opts = new GameOptions();
        opts.stopOnTurn = maxTurns;
        game.setGameOptions(opts);
        long t0 = System.currentTimeMillis();
        game.start(new Random().nextBoolean() ? pa.getId() : pb.getId());
        rec.snap(game, true);
        String w = game.getWinner();
        String res = w.contains("Player A ") ? "A" : w.contains("Player B ") ? "B" : "draw";
        return "{\"deckA\":" + q(Paths.get(a).getFileName().toString())
                + ",\"deckB\":" + q(Paths.get(b).getFileName().toString())
                + ",\"winner\":" + q(res) + ",\"turns\":" + game.getTurnNum()
                + ",\"secs\":" + (System.currentTimeMillis() - t0) / 1000.0
                + ",\"snapshots\":[" + String.join(",", rec.snaps) + "]}";
    }

    static RecordingAI addPlayer(Game game, Match match, String name, String deckPath, int skill,
                                 Recording rec) throws Exception {
        DeckCardLists list = DeckImporter.importDeckFromFile(deckPath, false);
        Deck deck = Deck.load(list, false, false);
        RecordingAI p = new RecordingAI(name, skill);
        p.rec = rec;
        game.loadCards(deck.getCards(), p.getId());
        game.addPlayer(p, deck);
        match.addPlayer(p, deck);
        return p;
    }
}
