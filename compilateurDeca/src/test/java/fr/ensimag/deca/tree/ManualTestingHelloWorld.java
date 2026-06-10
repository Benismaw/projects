package fr.ensimag.deca.tree;

import java.io.File;

import fr.ensimag.deca.CompilerOptions;
import fr.ensimag.deca.DecacCompiler;

public class ManualTestingHelloWorld {
    public static void main(String[] args) {
        try {
            // 1. Manually set up the file paths
            String source = "./src/test/deca/syntax/valid/provided/hello.deca";
            
            CompilerOptions options = new CompilerOptions();
            
            DecacCompiler compiler = new DecacCompiler(options, new File(source));
            
            boolean error = compiler.compile();
            
            if (error) {
                System.out.println("--- FAILED ---");
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}