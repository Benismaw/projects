package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.ARMInstruction;

public class ARMPrintString extends ARMInstruction {
    @Override
    public String toString() {
        return "bl __print_string";
    }
}
