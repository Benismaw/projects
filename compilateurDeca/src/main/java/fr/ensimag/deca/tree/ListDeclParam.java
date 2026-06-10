package fr.ensimag.deca.tree;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.RegisterOffset;
import fr.ensimag.deca.context.*;

import org.apache.log4j.Logger;

/**
 *
 * @author gl16
 * @date 01/01/2026
 */
public class ListDeclParam extends TreeList<AbstractDeclParam> {

    @Override
    public void decompile(IndentPrintStream s) {
        int i =0;
        for (AbstractDeclParam c : getList()) {
            if (i>0) s.print(", ");//la decompilation doit séparer les parametres par des virgules
            c.decompile(s);
            i++;
        }
    }

    //passe 2
    public Signature verifyListParam(DecacCompiler compiler) throws ContextualError {
        Signature sig=new Signature();
        for (AbstractDeclParam m : getList()) {
                Type typeParam=m.verifyDeclParam(compiler);
                sig.add(typeParam);
            }
            return sig;
        }
    //passe 3
    public void verifyListParamMembers(DecacCompiler compiler, EnvironmentExp localEnv) 
        throws ContextualError {
        int offset = 3; // On commence à 3 car -2 est pour 'this'
        for (AbstractDeclParam p : getList()) {
             // On transmet l'offset courant au paramètre
            p.verifyParamMembers(compiler, localEnv,offset);
            offset++;
        }
    }}
