package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.ARMInstruction;

public class ARMReadFloat extends ARMInstruction {
    @Override
    public String toString() {
        return "bl __read_float";
    }
}
