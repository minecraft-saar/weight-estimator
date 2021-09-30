# A weight estimator for irtg-based instruction giving

This system estimates grammar weights from interactions with the
instruction giving system.  The weights are either optimized to
predict the time it takes an instructee to carry out the instruction
or the number of errors the instructee will make when carrying out the
instruction.


Running minecraft-nlg with such a trained set of weights will either
minimize the expected time or the expected number of errors the
instructee makes.



Usage: WeightEstimator [-hV] [--sample-individual-instructions]
                       [--architect=<architectName>] [--db-pass=<dbPass>]
                       [--db-user=<dbUser>] [-l=<lowerPercentile>]
                       [--mode=<mode>] [-u=<upperPercentile>] <connStr>
Estimates grammar weights from played games.
      <connStr>            Connection String to connect to the DB
      --architect=<architectName>
                           Restrict to architect with this name. Leave empty
                             for no restriction.
      --db-pass=<dbPass>
      --db-user=<dbUser>
  -h, --help               Show this help message and exit.
  -l, --lower-percentile=<lowerPercentile>

      --mode=<mode>        OPTIMAL (default): LR on all data; BOOTSTRAPPED:
                             sample from bootstrapped LRs, BOTH: both
      --sample-individual-instructions
                           Bootstrap sampling on the instruction level instead
                             of the game level
  -u, --upper-percentile=<upperPercentile>

  -V, --version            Print version information and exit.
