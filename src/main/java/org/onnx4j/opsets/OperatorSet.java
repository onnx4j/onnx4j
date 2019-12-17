/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.opsets;

import java.util.Map;

/**
 * Each model MUST explicitly name the operator sets that it relies on for its
 * functionality. Operator sets define the available operators, their version,
 * and their status. Each model defines the imported operator sets by their
 * domains. All models implicitly import the default ONNX operator set.
 * 
 * Each operator set SHALL be defined in a separate document, also using
 * protobuf as the serialization format. How operator set documents are found at
 * runtime is implementation-dependent.
 * 
 * @author HarryLee
 *
 */
public abstract class OperatorSet extends OperatorSetId implements OperatorSetSpec {

	public static final String MAGIC = "ONNXOPSET";

	//
	// The ONNX version corresponding to the operators.
	//
	protected int irVersion;

	//
	// The prerelease component of the SemVer of the IR.
	//
	protected String irVersionPrerelease;

	//
	// The build metadata of this version of the operator set.
	//
	protected String irBuildMetadata;

	//
	// A human-readable documentation for this set of operators. Markdown is
	// allowed.
	//
	protected String docString;

	//
	// The operators of this operator set.
	//
	protected Map<String, Operator> operators;

	public OperatorSet(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(domain, opsetVersion);

		this.irVersion = irVersion;
		this.irVersionPrerelease = irVersionPrerelease;
		this.irBuildMetadata = irBuildMetadata;
		this.docString = docString;
		this.operators = this.initializeOperators();
	}

	public void registerOperator(Operator operator) {
		this.operators.put(operator.getOpType(), operator);
	}

	/* -- START OF GETTER -- */

	public String getId() {
		return super.id;
	}

	public int getIrVersion() {
		return irVersion;
	}

	public String getIrVersionPrerelease() {
		return irVersionPrerelease;
	}

	public String getIrBuildMetadata() {
		return irBuildMetadata;
	}

	public String getDomain() {
		return super.domain;
	}

	public long getOpsetVersion() {
		return super.opsetVersion;
	}

	public String getDocString() {
		return docString;
	}

	public Operator getOp(String opType) {
		return this.operators.get(opType);
	}

	/* -- END OF GETTER -- */

}