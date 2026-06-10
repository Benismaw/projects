package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.*;

public class ARMSvc extends ARMInstruction {

    private int value;

    public ARMSvc(int value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return "svc " + value;
    }
    
}
