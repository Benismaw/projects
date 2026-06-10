package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.ARMInstruction;

public class ARMPrintFloat extends ARMInstruction {
    @Override
    public String toString() {
        return "bl __print_float";
    }
}
