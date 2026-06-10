package fr.ensimag.deca;

import java.io.File;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import fr.ensimag.ima.pseudocode.Register;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

/**
 * User-specified options influencing the compilation.
 *
 * @author gl16
 * @date 01/01/2026
 */
public class CompilerOptions {
    public static final int QUIET = 0;
    public static final int INFO  = 1;
    public static final int DEBUG = 2;
    public static final int TRACE = 3;
    public int getDebug() {
        return debug;
    }

    public boolean getParse() {return parse; }

    public boolean getVerification() {return verification; }

    public boolean getNoCheck() {return noCheck; }

    public boolean getParallel() {
        return parallel;
    }

    public boolean getPrintBanner() {
        return printBanner;
    }

    public int getRegisters() {return registers; }

    public boolean getWarnings() { return warnings; }
    
    public List<File> getSourceFiles() {
        return Collections.unmodifiableList(sourceFiles);
    }

    private int debug = 0;
    private boolean parallel = false;
    private boolean printBanner = false;
    private boolean parse = false;
    private boolean verification= false;
    private boolean noCheck = false;
    private int registers = 16;
    private boolean warnings = false;
    private List<File> sourceFiles = new ArrayList<File>();
    private boolean arm = false;

    public boolean isARM() {
        return arm;
    }

    
    public void parseArgs(String[] args) throws CLIException {
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            switch(arg) {
                case "-b":
                    if (args.length > 1) {
                        throw new CLIException("L'option -b doit être utilisée seule.");
                    }
                    printBanner = true;
                    break;
                case "-p":
                    parse = true;
                    break;
                case "-v":
                    verification = true;
                    break;
                case "-n":
                    noCheck = true;
                    break;
                case "-P":
                    parallel = true;
                    break;
                case "-w":
                    warnings = true;
                    break;
                case "-d":
                    debug ++;
                    break;
                case "-r":
                    // On verifie qu'il y a un argument apres
                    if(i+1 >= args.length) {
                        throw new CLIException("-r attend un nombre de registres");
                    }
                    try {
                        registers = Integer.parseInt(args[i+1]);
                        i++;
                    } catch (NumberFormatException e) {
                        throw new CLIException("Nombres de registres incorrectes");
                    }
                    if (registers < 4 || registers > 16) {
                        throw new CLIException("Le nombre de registres doit être entre 4 et 16");
                    }
                    break;
                case "-arm":
                    arm = true;
                    break;
                default:
                    if (arg.endsWith(".deca")) {
                        File f = new File(arg);
                        // Si un fichier apparaît plusieurs fois, il n’est compilé qu’une seule fois
                        if (!sourceFiles.contains(f)) {
                            sourceFiles.add(f);
                        }
                    } else {
                        throw new CLIException("Option ou fichier invalide : " + arg);
                    }
                    break;
            }

        }
        // Verification des imcompatibilités
        if (parse && verification) {
            throw new CLIException("Les options -p et -v sont incompatibles.");
        }
        if (printBanner && !sourceFiles.isEmpty()) {
            throw new CLIException("L'option -b ne prend pas de fichier source.");
        }

        Logger logger = Logger.getRootLogger();
        // map command-line debug option to log4j's level.
        switch (getDebug()) {
            case QUIET: break; // keep default
            case INFO:
                logger.setLevel(Level.INFO); break;
            case DEBUG:
                logger.setLevel(Level.DEBUG); break;
            case TRACE:
                logger.setLevel(Level.TRACE); break;
            default:
                logger.setLevel(Level.ALL); break;
        }
        logger.info("Application-wide trace level set to " + logger.getLevel());

        boolean assertsEnabled = false;
        assert assertsEnabled = true; // Intentional side effect!!!
        if (assertsEnabled) {
            logger.info("Java assertions enabled");
        } else {
            logger.info("Java assertions disabled");
        }
    }

    protected void displayUsage() {
        System.out.println("Usage: decac [[-p | -v] [-n] [-r X] [-d]* [-P] [-w] <fichier deca>...] | [-b]");
        System.out.println("  -b       (banner)       : Affiche la bannière (doit être seule)");
        System.out.println("  -p       (parse)        : Arrête decac après construction de l'arbre");
        System.out.println("  -v       (verification) : Arrête decac après vérifications");
        System.out.println("  -n       (no check)     : Supprime les tests de débordement");
        System.out.println("  -r X     (registers)    : Limite les registres à R0 ... R{X-1}");
        System.out.println("  -d       (debug)        : Active les traces de debug");
        System.out.println("  -P       (parallel)     : Active la compilation parallèle");
        System.out.println("  -w       (warnings)     : Active les avertissements");
        System.out.println("  -arm                    : Génère du code ARM");
    }
}
