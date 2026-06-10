package fr.ensimag.ARM;

import java.io.PrintStream;

/**
 * Portion of IMA assembly code to be dumped verbatim into the
 * generated code.
 *
 * @author Ensimag
 * @date 01/01/2026
 */
public class ARMInlinePortion extends ARMAbstractLine {
    private final String asmCode;
    
    public ARMInlinePortion(String asmCode) {
        super();
        this.asmCode = asmCode;
    }
    
    @Override
    void display(PrintStream s) {
        s.println(asmCode);
    }

}
