#!/bin/sh
cd "$(dirname "$0")"
java $MZ_JAVA_OPTS -Xms1g -Xmx${MZ_XMX:-11g} -XX:+UseParallelGC --add-opens=java.base/java.lang=ALL-UNNAMED -jar lib/mage-magezero-1.4.63.jar "$@"