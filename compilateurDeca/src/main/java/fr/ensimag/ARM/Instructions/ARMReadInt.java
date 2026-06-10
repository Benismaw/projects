package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.ARMInstruction;

public class ARMReadInt extends ARMInstruction {
    @Override
    public String toString() {
        return "bl __read_int";
    }
}
