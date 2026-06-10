package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.ARMInstruction;
import fr.ensimag.ARM.ARMOperand;
import fr.ensimag.ARM.ARMRegister;

public class ARMStr extends ARMInstruction {

    private final ARMRegister source;
    private final ARMOperand address;

    public ARMStr(ARMRegister source, ARMOperand address) {
        this.source = source;
        this.address = address;
    }

    @Override
    public String toString() {
        return "str " + source + ", [" + address + "]";
    }

}
