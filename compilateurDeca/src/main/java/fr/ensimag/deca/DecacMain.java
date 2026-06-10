package fr.ensimag.deca;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

import org.apache.log4j.Logger;

/**
 * Main class for the command-line Deca compiler.
 *
 * @author gl16
 * @date 01/01/2026
 */
public class DecacMain {
    private static Logger LOG = Logger.getLogger(DecacMain.class);
    
    public static void main(String[] args) {
        // example log4j message.
        LOG.info("Decac compiler started");
        boolean error = false;
        final CompilerOptions options = new CompilerOptions();
        try {
            options.parseArgs(args);
        } catch (CLIException e) {
            System.err.println("Error during option parsing:\n"
                    + e.getMessage());
            options.displayUsage();
            System.exit(1);
        }
        if (options.getPrintBanner()) {
            System.out.println("-------------------------------------");
            System.out.println("--------GL16 Compilateur Deca--------");
            System.out.println("-------------------------------------");
            System.exit(0);
        }
        if (options.getSourceFiles().isEmpty()) {
            options.displayUsage();
            System.exit(1);
        }
        if (options.getParallel()) {
            int cores = Runtime.getRuntime().availableProcessors();
            ExecutorService executor = Executors.newFixedThreadPool(cores);
            List<Callable<Boolean>> tasks = new ArrayList<>();
            for (File f : options.getSourceFiles()) {
                tasks.add(() -> {
                    DecacCompiler compiler = new DecacCompiler(options, f);
                    return compiler.compile();
                });
            }
            try {
                // On lance tout
                List<Future<Boolean>> results = executor.invokeAll(tasks);

                for (Future<Boolean> result : results) {
                    if (result.get()) {  // True --> erreur
                        error = true;
                    }
                }

            } catch (Exception e) {
                System.err.println("Erreur durant la compilation parallèle : " + e.getMessage());
                error = true;
            }

        } else {
            for (File source : options.getSourceFiles()) {
                DecacCompiler compiler = new DecacCompiler(options, source);
                if (compiler.compile()) {
                    error = true;
                }
            }
        }
        System.exit(error ? 1 : 0);
    }
}
