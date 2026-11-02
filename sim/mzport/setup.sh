#!/bin/sh
# Rebuild the MageZero self-play workspace from this repo: toolchain, our XMage fork, MageZero's Python side,
# the runner bundle, the deck pool, and (optionally) a data snapshot made by snapshot.sh.
#
#   sim/mzport/setup.sh [snapshot.tar]
#
# Workspace: $LMT_SIM (default ~/pers/lmt-magic-sim). Re-running skips whatever already exists.
# Needs: git, curl, unzip, python3 (3.12+), an NVIDIA driver for the GPU server. ~15 GB disk, ~20 min.
set -eu

REPO=$(cd "$(dirname "$0")/../.." && pwd)
HERE=$REPO/sim/mzport
SIM=${LMT_SIM:-$HOME/pers/lmt-magic-sim}
MZ=$SIM/mzport
XMAGE_BASE=6b07e46d8de1e57bdab5d8852a6e6e8a78b5602c     # upstream magefree/mage our patches apply to
MAGEZERO_BASE=fb6ee21                                    # WillWroble/MageZero main our patch applies to
mkdir -p $SIM/xm $MZ

echo "== toolchain (Temurin JDK 21, Maven 3.9.9) in $SIM/xm"
if [ ! -x $SIM/xm/jdk/bin/java ]; then
  curl -fL -o $SIM/xm/jdk21.tgz "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse"
  mkdir -p $SIM/xm/jdk && tar xzf $SIM/xm/jdk21.tgz -C $SIM/xm/jdk --strip-components=1
fi
if [ ! -x $SIM/xm/apache-maven-3.9.9/bin/mvn ]; then
  curl -fL -o $SIM/xm/maven.tgz https://archive.apache.org/dist/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.tar.gz
  tar xzf $SIM/xm/maven.tgz -C $SIM/xm
fi
export JAVA_HOME=$SIM/xm/jdk
MVN="$SIM/xm/apache-maven-3.9.9/bin/mvn -Dmaven.repo.local=$MZ/m2 -Djava.version=17 -Djava.release=17"

echo "== XMage fork: upstream $XMAGE_BASE + $(ls $HERE/mage-patches/*.patch | wc -l) patches (branch lmt-ai-port)"
if [ ! -d $MZ/mage/.git ]; then
  git clone --filter=blob:none https://github.com/magefree/mage.git $MZ/mage
  git -C $MZ/mage checkout -q -b lmt-ai-port $XMAGE_BASE
  git -C $MZ/mage -c user.name=lmt -c user.email=lmt@local am -q $HERE/mage-patches/*.patch
fi
if [ ! -f $MZ/mage/Mage.MageZero/target/mage-magezero.zip ]; then
  (cd $MZ/mage && MAVEN_OPTS=-Xmx6g $MVN -q -DskipTests install -pl Mage.MageZero -am)
fi

echo "== MageZero (Python): $MAGEZERO_BASE + magezero.patch"
if [ ! -d $MZ/magezero-py/.git ]; then
  git clone https://github.com/WillWroble/MageZero.git $MZ/magezero-py
  git -C $MZ/magezero-py checkout -q $MAGEZERO_BASE
  git -C $MZ/magezero-py apply $HERE/magezero.patch
fi

echo "== runner bundle in magezero-py/xmage"
mkdir -p $MZ/magezero-py/xmage
(cd $MZ/magezero-py/xmage && unzip -q -o $MZ/mage/Mage.MageZero/target/mage-magezero.zip 'lib/*')
cp $HERE/mz-xmage.sh $MZ/magezero-py/xmage/mz-xmage.sh
# the bundled launcher calls plain `java`: put ours first
sed -i "s|^java |$SIM/xm/jdk/bin/java |" $MZ/magezero-py/xmage/mz-xmage.sh
chmod +x $MZ/magezero-py/xmage/mz-xmage.sh

echo "== Python env for the model server and trainer: $SIM/.venv-ml"
if [ ! -x $SIM/.venv-ml/bin/python ]; then
  python3 -m venv $SIM/.venv-ml
  $SIM/.venv-ml/bin/pip install -q --index-url https://download.pytorch.org/whl/cu128 torch
  $SIM/.venv-ml/bin/pip install -q -e $MZ/magezero-py py-spy
fi

echo "== deck pool (2001 FRA sealed decks the generations draw pairs from)"
[ -d $SIM/simdecks ] || tar xzf $HERE/simdecks.tgz -C $SIM

if [ $# -ge 1 ]; then
  echo "== data snapshot $1"
  tar xf "$1" -C $MZ/magezero-py
fi

mkdir -p $SIM/selfplay_view/games
cat <<EOF

Done. Workspace: $SIM
  continue the loop (a finished generation is skipped, so it resumes where the snapshot ended):
    cd $REPO && LMT_SIM=$SIM $SIM/.venv-ml/bin/python scripts/mz_loop.py --gens 6 --games 120 --budget 150
  viewer on :8743:
    LMT_SIM=$SIM $HERE/selfplay_refresh.sh &   (cd $SIM/selfplay_view && python3 -m http.server 8743)
See $HERE/README.md.
EOF
