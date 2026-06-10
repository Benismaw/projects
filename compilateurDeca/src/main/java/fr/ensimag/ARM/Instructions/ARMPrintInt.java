package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.ARMInstruction;

public class ARMPrintInt extends ARMInstruction {
    @Override
    public String toString() {
        return "bl __print_int";
    }
}
