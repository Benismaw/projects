package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.ARMInstruction;

public class ARMDiv extends ARMInstruction {
    @Override
    public String toString() {
        return "bl __div";
    }
}
