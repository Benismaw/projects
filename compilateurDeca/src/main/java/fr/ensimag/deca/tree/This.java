package fr.ensimag.deca.tree;

import java.io.PrintStream;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.ARMRegisterOffset;
import fr.ensimag.ARM.Instructions.ARMLdr;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.RegisterOffset;
import fr.ensimag.ima.pseudocode.instructions.LOAD;

public class This extends AbstractExpr {

    private boolean implicit;

    public This(boolean implicit) {
        this.implicit = implicit;
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
                           ClassDefinition currentClass) throws ContextualError {
        // Règle 3.43
        if (currentClass == null) {
            throw new ContextualError("Utilisation de 'this' interdite dans le programme principal (règle: 3.43)", getLocation());
        }
        Type type = currentClass.getType();
        setType(type);
        return type;
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        // L'adresse de 'this' est fixée à -2(LB) par convention de liaison
        compiler.addInstruction(new LOAD(new RegisterOffset(-2,Register.LB),register));
    }

    @Override
    public void decompile(IndentPrintStream s) {
        if (!implicit) {
            s.print("this");
        }
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        // Feuille
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        // Feuille
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        var prog = compiler.getARMProgram();

        // L'adresse de 'this' est fixée à -2(LB) par convention de liaison
        prog.addInstruction(new ARMLdr(register, new ARMRegisterOffset(-2, ARMRegister.R11)));
    }
}